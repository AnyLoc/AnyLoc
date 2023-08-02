# Show VLAD clustering for set of example images or a user image
"""
    User input:
    - Domain: Indoor, Aerial, or Urban
    - Image: Image to be clustered
    - Cluster numbers (to visualize)
    - Pixel coordinates (to pick further clusters)
    - A unique cache ID (to store the DINO forward passes)
    
    There are example images for each domain.
    
    Output:
    - All images with cluster assignments
    
    Some Gradio links:
    - Controlling layout
        - https://www.gradio.app/guides/quickstart#blocks-more-flexibility-and-control
    - Data state (persistence)
        - https://www.gradio.app/guides/interface-state
        - https://www.gradio.app/docs/state
    - Layout control
        - https://www.gradio.app/guides/controlling-layout
        - https://www.gradio.app/guides/blocks-and-event-listeners
"""

# %%
import os
import gradio as gr
import numpy as np
import cv2 as cv
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as tvf
from torchvision.transforms import functional as T
from PIL import Image
import matplotlib.pyplot as plt
import distinctipy as dipy
from typing import Literal, List
import gradio as gr
import time
import glob
import shutil
from copy import deepcopy
# DINOv2 imports
from utilities import DinoV2ExtractFeatures
from utilities import VLAD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %% Global Variables (for program)
# Realpath expansion
_ex = lambda x: os.path.realpath(os.path.expanduser(x))
# Folder where the cache for this app is stored
cache_dir: str = _ex("./cache")
assert os.path.isdir(cache_dir), "Cache directory not found"
# Maximum image dimension
max_img_size: int = 1024
# Maximum number of images to upload
max_num_imgs: int = 10


# %%
# Types
T1 = Literal["query", "key", "value", "token"]
T2 = Literal["aerial", "indoor", "urban"]
DOMAINS = ["aerial", "indoor", "urban"]
T3 = Literal["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", 
                "dinov2_vitg14"]
# Main Gradio application
class DINOv2GradioApp:
    # Constructor
    def __init__(self, dino_model: T3 = "dinov2_vitg14", 
                desc_layer: int = 31, desc_facet: T1 = "value",
                num_c: int = 8) -> None:
        # DINO extractor (shared across all)
        print("Loading DINO model")
        self.extractor = DinoV2ExtractFeatures(dino_model, desc_layer,
                desc_facet, device=device)
        print("DINO model loaded")
        # VLAD path (directory)
        self.num_c = num_c
        ext_s = f"{dino_model}/l{desc_layer}_{desc_facet}_c{num_c}"
        self.vc_dir = os.path.join(cache_dir, "vocabulary", ext_s)
        # Base image transformations
        self.base_tf = tvf.Compose([
            tvf.ToTensor(),
            tvf.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        ])
        # The demo app
        self.demo: gr.Blocks = None
    
    def get_vlad_clusters(self, domain: T2, pr = gr.Progress()):
        dm = str(domain).lower()
        assert dm in DOMAINS, "Invalid domain"
        # Load VLAD cluster centers
        pr(0, desc="Loading VLAD clusters")
        c_centers_file = os.path.join(self.vc_dir, dm, 
                "c_centers.pt")
        if not os.path.isfile(c_centers_file):
            return f"Cluster centers not found for: {domain}"
        c_centers = torch.load(c_centers_file)
        pr(0.5)
        num_c = c_centers.shape[0]
        desc_dim = c_centers.shape[1]
        vlad = VLAD(num_c, desc_dim, 
                cache_dir=os.path.dirname(c_centers_file))
        vlad.fit(None)  # Restore the cache
        pr(1)
        return f"VLAD clusters loaded for: {domain}", vlad
    
    def var_num_img(self, s):
        n = int(s)  # Slider value as int
        return [gr.Image.update(label=f"Image {i+1}", visible=True) \
                for i in range(n)] + [gr.Image.update(visible=False) \
                        for _ in range(max_num_imgs - n)]
    
    @torch.no_grad()
    def get_descs(self, imgs_batch: List[np.ndarray], 
                pr = gr.Progress()):
        pr(0, desc="Extracting descriptors")
        patch_descs = []
        for i, img in enumerate(imgs_batch):
            # Convert to PIL image
            pil_img = Image.fromarray(img)
            img_pt = self.base_tf(pil_img).to(device)
            if max(img_pt.shape[-2:]) > max_img_size:
                print(f"Image {i+1}: {img_pt.shape[-2:]}, outside")
                c, h, w = img_pt.shape
                # Maintain aspect ratio
                if h == max(img_pt.shape[-2:]):
                    w = int(w * max_img_size / h)
                    h = max_img_size
                else:
                    h = int(h * max_img_size / w)
                    w = max_img_size
                img_pt = T.resize(img_pt, (h, w), 
                    interpolation=T.InterpolationMode.BICUBIC)
                pil_img = pil_img.resize((w, h))    # Backup
            # Make image patchable
            c, h, w = img_pt.shape
            h_new, w_new = (h // 14) * 14, (w // 14) * 14
            img_pt = tvf.CenterCrop((h_new, w_new))(img_pt)[None, ...]
            # Extract descriptors
            ret = self.extractor(img_pt).cpu()  # [1, n_p, d]
            patch_descs.append({"img": pil_img, "descs": ret})
            pr((i+1) / len(imgs_batch))
        return patch_descs, \
                f"Descriptors extracted for {len(imgs_batch)} images"
    
    def assign_vlad(self, patch_descs, vlad: VLAD, 
                pr = gr.Progress()):
        img_patch_descs = [pd["descs"] for pd in patch_descs]
        pr(0, desc="Assigning VLAD clusters")
        desc_assignments = []   # List[Tensor;shape=('h', 'w');int]
        for i, qu_desc in enumerate(img_patch_descs):
            # Residual vectors; 'n' could differ (based on img sizes)
            res = vlad.generate_res_vec(qu_desc[0]) # ['n', n_c, d]
            img = patch_descs[i]["img"]
            h, w, c = np.array(img).shape
            h_p, w_p = h // 14, w // 14
            h_new, w_new = h_p * 14, w_p * 14
            assert h_p * w_p == res.shape[0], "Residual incorrect!"
            # Descriptor assignments
            da = res.abs().sum(dim=2).argmin(dim=1).reshape(h_p, w_p)
            da = F.interpolate(da[None, None, ...].to(float),
                    (h_new, w_new), mode="nearest")[0, 0].to(da.dtype)
            desc_assignments.append(da)
            pr((i+1) / len(img_patch_descs))
        pr(1.0)
        return desc_assignments, "VLAD clusters assigned"
    
    def get_ca_images(self, desc_assignments, patch_descs, alpha,
                pr = gr.Progress()):
        if desc_assignments is None or len(desc_assignments) == 0:
            return None, "First load images"
        c_colors = dipy.get_colors(self.num_c, rng=928, 
                colorblind_type="Deuteranomaly")
        np_colors = (np.array(c_colors) * 255).astype(np.uint8)
        # Get images with clusters
        pil_imgs = [pd["img"] for pd in patch_descs]
        res_imgs = []   # List[PIL.Image]
        pr(0, desc="Generating cluster assignment images")
        for i, pil_img in enumerate(pil_imgs):
            # Descriptor assignment image: [h, w, 3]
            da: torch.Tensor = desc_assignments[i]    # ['h', 'w']
            da_img = np.zeros((*da.shape, 3), dtype=np.uint8)
            for c in range(self.num_c):
                da_img[da == c] = np_colors[c]
            # Background image: [h, w, 3]
            img_np = np.array(pil_img, dtype=np.uint8)
            h, w, c = np.array(img_np).shape
            h_p, w_p = (h // 14), (w // 14)
            h_new, w_new = h_p * 14, w_p * 14
            img_np = F.interpolate(torch.tensor(img_np)\
                    .permute(2, 0, 1)[None, ...], (h_new, w_new),
                    mode='nearest')[0].permute(1, 2, 0).numpy()
            res_img = cv.addWeighted(img_np, 1 - alpha, da_img, alpha,
                                        0.0)
            res_imgs.append(Image.fromarray(res_img))
            pr((i+1) / len(pil_imgs))
        pr(1.0)
        return res_imgs, "Cluster assignment images generated"
    
    # Build the UI (interface)
    def build_interface(self):
        with gr.Blocks() as self.demo:
            # Domain selection (for VLAD cluster centers)
            d_vals = [k.title() for k in DOMAINS]
            domain = gr.Radio(d_vals, value=d_vals[0])
            # Add images
            nimg_s = gr.Slider(1, max_num_imgs, value=1, step=1, 
                    label="How many images?")
            with gr.Row():
                imgs = [gr.Image(label=f"Image {i+1}", visible=True) \
                        for i in range(nimg_s.value)] + \
                        [gr.Image(visible=False) \
                        for _ in range(max_num_imgs - nimg_s.value)]
                for i, img in enumerate(imgs):
                    img.change(lambda _: None, img)
            with gr.Row():
                imgs2 = [gr.Image(label=f"VLAD Clusters {i+1}", 
                        visible=False) for i in range(max_num_imgs)]
            nimg_s.change(lambda s: self.var_num_img(s), nimg_s, imgs)
            # Cluster center
            blend_alpha = gr.Slider(0, 1, 0.4, step=0.01, 
                    label="Blend alpha (weight for cluster centers)")
            # Part 1: Show the cluster images
            bttn1 = gr.Button("Click Me!")  # Upto cluster assignment
            # State declarations
            vlad = gr.State()   # VLAD object
            desc_assignments = gr.State()   # Cluster assignments
            imgs_batch = gr.State() # Images as batch
            # A wrapper to batch the images
            def batch_images(data):
                sv = data[nimg_s]
                images: List[np.ndarray] = [data[imgs[k]] \
                        for k in range(sv)]
                return images
            # A wrapper to unbatch images (and pad to max)
            def unbatch_images(imgs_batch):
                ret = [gr.Image.update(visible=False) \
                        for _ in range(max_num_imgs)]
                if imgs_batch is None or len(imgs_batch) == 0:
                    return ret
                for i, img_pil in enumerate(imgs_batch):
                    img_np = np.array(img_pil)
                    ret[i] = gr.Image.update(img_np, visible=True)
                return ret
            # A state to store descriptors
            patch_descs = gr.State()
            out_msg1 = gr.Markdown("Select domain and upload images")
            out_msg2 = gr.Markdown("For descriptor extraction")
            out_msg3 = gr.Markdown("Followed by VLAD assignment")
            out_msg4 = gr.Markdown("Followed by cluster images")
            # Get the VLAD images on button click (callback)
            bttn1.click( # Get VLAD object (loaded cluster centers)
                    lambda d: self.get_vlad_clusters(d), 
                    domain,
                    [out_msg1, vlad])\
                .then(  # Get List[np.ndarray] for images
                    batch_images, 
                    {nimg_s, *imgs, imgs_batch},
                    imgs_batch)\
                .then(  # Get the descriptors
                    lambda imgs: self.get_descs(imgs),
                    imgs_batch,
                    [patch_descs, out_msg2])\
                .then(  # Get VLAD cluster (assignments) per image
                    lambda pd, vl: self.assign_vlad(pd, vl),
                    [patch_descs, vlad],
                    [desc_assignments, out_msg3])\
                .then(  # Get cluster assignment images
                    lambda das, pds, al: \
                        self.get_ca_images(das, pds, al),
                    [desc_assignments, patch_descs, blend_alpha],
                    [imgs_batch, out_msg4])\
                .then(  # Unbatch the images
                    unbatch_images,
                    imgs_batch,
                    imgs2)
            blend_alpha.change(  # Update the cluster images
                    lambda das, pds, al: \
                        self.get_ca_images(das, pds, al),
                    [desc_assignments, patch_descs, blend_alpha],
                    [imgs_batch, out_msg4])\
                .then(  # Unbatch the images
                    unbatch_images,
                    imgs_batch,
                    imgs2)
    
    # Deploy the UI
    def deploy(self, share=False):
        self.demo.queue().launch(share=share)
    
    # Build and deploy the UI
    def build_and_deploy(self, share=False):
        self.build_interface()
        print("Interface build completed")
        self.deploy(share=share)
        print("Application deployment completed")


# %%
if __name__ == "__main__":
    # Check if everything exists
    assert os.path.isdir(cache_dir), "Cache directory not found"
    # Initialize app
    app = DINOv2GradioApp()
    print("Loaded the application")
    app.build_and_deploy(share=False)
    print("Ended!")

