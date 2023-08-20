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

# A markdown string shown at the top of the app
header_markdown = """
# AnyLoc Demo

\|  [Website](https://anyloc.github.io/) \| \
    [GitHub](https://github.com/AnyLoc/AnyLoc) \| \
    [YouTube](https://youtu.be/ITo8rMInatk) \|


This space contains a collection of demos for AnyLoc. Each demo is a \
self-contained application in the tabs below. The following \
applications are included

1. **GeM t-SNE Projection**: Upload a set of images and see where \
    they land on a t-SNE projection of GeM descriptors from many \
    domains. This can be used to guide domain selection (from a few \
    representative images).
2. **Cluster Visualization**: This visualizes the VLAD cluster \
    assignments for the patch descriptors. You need to select the \
    domain for loading VLAD cluster centers (vocabulary).

We do **not** save any images uploaded to the demo. Some errors may \
leave a log. We do not collect any information about the user. The \
example images are attributed in the respective tabs.

🥳 Thanks to HuggingFace for providing a free GPU for this demo.

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
from sklearn.manifold import TSNE
import distinctipy as dipy
import joblib
from typing import Literal, List
import gradio as gr
import time
import glob
import shutil
import matplotlib.pyplot as plt
from copy import deepcopy
# DINOv2 imports
from utilities import DinoV2ExtractFeatures
from utilities import VLAD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
# Configurations
T1 = Literal["query", "key", "value", "token"]
T2 = Literal["aerial", "indoor", "urban"]
DOMAINS = ["aerial", "indoor", "urban"]
T3 = Literal["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", 
                "dinov2_vitg14"]
_ex = lambda x: os.path.realpath(os.path.expanduser(x))
dino_model: T3 = "dinov2_vitg14"
desc_layer: int = 31
desc_facet: T1 = "value"
num_c: int = 8
cache_dir: str = _ex("./cache") # Directory containing program cache
max_img_size: int = 1024    # Image resolution (max dim/size)
max_num_imgs: int = 16      # Max number of images to upload
share: bool = False          # Share application using .gradio link

# Verify inputs
assert os.path.isdir(cache_dir), "Cache directory not found"


# %%
# Model and transforms
print("Loading DINO model")
# extractor = None  # FIXME: For quick testing only
extractor = DinoV2ExtractFeatures(dino_model, desc_layer, desc_facet, 
                                    device=device)
print("DINO model loaded")
# VLAD path (directory)
ext_s = f"{dino_model}/l{desc_layer}_{desc_facet}_c{num_c}"
vc_dir = os.path.join(cache_dir, "vocabulary", ext_s)
assert os.path.isdir(vc_dir), f"VLAD directory: {vc_dir} not found"
# GeM path (cache)
gem_cf = os.path.join(cache_dir, "gem_cache", "result_dino_v2.gz")
assert os.path.isfile(gem_cf), f"GeM cache: {gem_cf} not found"
gem_cache = joblib.load(gem_cf)
assert gem_cache["model"]["type"] == dino_model
assert gem_cache["model"]["layer"] == desc_layer
assert gem_cache["model"]["facet"] == desc_facet
fig = plt.figure()  # Main figure
fig.clear()
# Base image transformations
base_tf = tvf.Compose([
    tvf.ToTensor(),
    tvf.Normalize(mean=[0.485, 0.456, 0.406], 
                    std=[0.229, 0.224, 0.225])
])


# %%
# Get VLAD object
def get_vlad_clusters(domain, pr = gr.Progress()):
    dm: T2 = str(domain).lower()
    assert dm in DOMAINS, "Invalid domain"
    # Load VLAD cluster centers
    pr(0, desc="Loading VLAD clusters")
    c_centers_file = os.path.join(vc_dir, dm, "c_centers.pt")
    if not os.path.isfile(c_centers_file):
        return f"Cluster centers not found for: {domain}", None
    c_centers = torch.load(c_centers_file)
    pr(0.5)
    num_c = c_centers.shape[0]
    desc_dim = c_centers.shape[1]
    vlad = VLAD(num_c, desc_dim, 
            cache_dir=os.path.dirname(c_centers_file))
    vlad.fit(None)  # Restore the cache
    pr(1)
    return f"VLAD clusters loaded for: {domain}", vlad


# %%
# Get VLAD descriptors
@torch.no_grad()
def get_descs(imgs_batch, pr = gr.Progress()):
    imgs_batch: List[np.ndarray] = imgs_batch
    pr(0, desc="Extracting descriptors")
    patch_descs = []
    for i, img in enumerate(imgs_batch):
        if img is None:
            print(f"Image {i+1} is None")
            continue
        # Convert to PIL image
        pil_img = Image.fromarray(img)
        img_pt = base_tf(pil_img).to(device)
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
        ret = extractor(img_pt).cpu()  # [1, n_p, d]
        patch_descs.append({"img": pil_img, "descs": ret})
        pr((i+1) / len(imgs_batch))
    pr(1.0)
    return patch_descs, \
            f"Descriptors extracted for {len(imgs_batch)} images"


# %%
# Assign VLAD clusters (descriptor assignment)
def assign_vlad(patch_descs, vlad, pr = gr.Progress()):
    vlad: VLAD = vlad
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


# %%
# Cluster assignments to images
def get_ca_images(desc_assignments, patch_descs, alpha,
            pr = gr.Progress()):
    if desc_assignments is None or len(desc_assignments) == 0:
        if not 0 <= alpha <= 1:
            return None, f"Invalid alpha value: {alpha} (should be "\
                    "between 0 and 1)"
        return None, "First load the images"
    c_colors = dipy.get_colors(num_c, rng=928, 
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
        for c in range(num_c):
            da_img[da == c] = np_colors[c]
        # Background image: [h, w, 3]
        img_np = np.array(pil_img, dtype=np.uint8)
        h, w, c = np.array(img_np).shape
        h_p, w_p = (h // 14), (w // 14)
        h_new, w_new = h_p * 14, w_p * 14
        img_np = F.interpolate(torch.tensor(img_np)\
                .permute(2, 0, 1)[None, ...], (h_new, w_new),
                mode='nearest')[0].permute(1, 2, 0).numpy()
        res_img = cv.addWeighted(img_np, 1 - alpha, da_img, alpha, 0.)
        res_imgs.append(Image.fromarray(res_img))
        pr((i+1) / len(pil_imgs))
    pr(1.0)
    return res_imgs, "Cluster assignment images generated"


# %%
# Get GeM descriptors from cache
def get_gem_descs_cache(use_d, pr = gr.Progress()):
    use_d: List[str] = use_d
    if len(use_d) == 0:
        return "Select at least one domain", None
    else:
        use_d = [d.lower() for d in use_d]
    indoor_datasets = ["baidu_datasets", "gardens", "17places"]
    urban_datasets = ["pitts30k", "st_lucia", "Oxford"]
    aerial_datasets = ["Tartan_GNSS_test_rotated", 
            "Tartan_GNSS_test_notrotated", "VPAir"]
    pr(0, desc="Loading GeM descriptors from cache")
    gem_descs = {
        "labels": [],
        "descs": [],
    }
    for i, ds in enumerate(gem_cache["data"]):
        # GeM descriptors from data: n_desc, desc_dim
        d: np.ndarray = gem_cache["data"][ds]["descriptors"]
        if ds in indoor_datasets and "indoor" in use_d:
            gem_descs["labels"].extend(["indoor"] * d.shape[0])
        elif ds in urban_datasets and "urban" in use_d:
            gem_descs["labels"].extend(["urban"] * d.shape[0])
        elif ds in aerial_datasets and "aerial" in use_d:
            gem_descs["labels"].extend(["aerial"] * d.shape[0])
        else:
            continue
        gem_descs["descs"].append(d)
        pr((i+1) / len(gem_cache["data"]))
    gem_descs["descs"] = np.concatenate(gem_descs["descs"], axis=0)
    pr(1.0)
    return "GeM descriptors loaded from cache", gem_descs


# %%
# Get GeM pooled features of the uploaded images
def get_add_gem_descs(imgs_batch, gem_descs, pr = gr.Progress()):
    imgs_batch: List[np.ndarray] = imgs_batch
    gem_descs: dict = gem_descs
    pr(0, desc="Extracting GeM descriptors")
    num_imgs_extracted = 0
    for i, img in enumerate(imgs_batch):
        if img is None:
            print(f"Image {i+1} is None")
            continue
        # Convert to PIL image
        pil_img = Image.fromarray(img)
        img_pt = base_tf(pil_img).to(device)
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
        ret = extractor(img_pt).cpu()  # [1, n_p, d]
        # Get the GeM pooled descriptor
        x = torch.mean(ret**3, dim=-2)
        g_res = x.to(torch.complex64) ** (1/3)
        g_res = torch.abs(g_res) * torch.sign(x)    # [1, d]
        g_res = g_res.numpy()
        # Add to state
        gem_descs["labels"].append(f"Image{i+1}")
        gem_descs["descs"] = np.concatenate([gem_descs["descs"], 
                                            g_res])
        num_imgs_extracted += 1
        pr((i+1) / len(imgs_batch))
    pr(1.0)
    gem_descs["num_uimgs"] = num_imgs_extracted
    return gem_descs, "GeM descriptors extracted"


# %%
# Apply tSNE to the GeM descriptors
def get_tsne_fm_gem(gem_descs, pr = gr.Progress()):
    pr(0, desc="Applying tSNE to GeM descriptors")
    desc_all: np.ndarray = gem_descs["descs"]   # [n, d_dim]
    labels_all: List[str] = gem_descs["labels"] # [n]
    # tSNE projection
    tsne = TSNE(n_components=2, random_state=30, perplexity=50, 
            learning_rate=200, init='random')
    desc_2d = tsne.fit_transform(desc_all)
    # Result
    tsne_pts = {
        "labels": labels_all,
        "pts": desc_2d,
        "num_uimgs": gem_descs["num_uimgs"],    # Number of user imgs
    }
    pr(1.0)
    return tsne_pts, "tSNE projection done"


# %%
# Plot tSNE to matplotlib figure
def plot_tsne(tsne_pts):
    colors = {
        "aerial": (80/255,  0/255,  80/255),
        "indoor": ( 0/255, 76/255, 204/255),
        "urban":  ( 0/255, 204/255,  0/255),
    }
    ni = int(tsne_pts["num_uimgs"])
    # Custom colors for user images
    ucs = dipy.get_colors(ni, exclude_colors=list(colors.values())\
            .extend([(0, 0, 0), (1, 1, 1)]), 
            colorblind_type="Deuteranomaly")
    for i in range(ni):
        colors[f"Image{i+1}"] = ucs[i]
    fig.clear()
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("tSNE Projection")
    for i, domain in enumerate(list(colors.keys())):
        pts = tsne_pts["pts"][np.array(tsne_pts["labels"]) == domain]
        if domain.startswith("Image"):
            m = "x"
        else:
            m = "o"
        ax.scatter(pts[:, 0], pts[:, 1], label=domain, marker=m,
                color=colors[domain])
    # Put legend at the bottom of axis
    ax.legend()
    ax.set_xticks([])
    ax.set_yticks([])
    fig.set_tight_layout(True)
    # fig.set_tight_layout(True)
    return fig, "tSNE plot created"


# %%
print("Interface build started")


# Tab for VLAD cluster assignment visualization
def tab_cluster_viz():
    d_vals = [k.title() for k in DOMAINS]
    domain = gr.Radio(d_vals, value=d_vals[0], label="Domain",
            info="The domain of images (for loading VLAD vocabulary)")
    nimg_s = gr.Number(2, label="How many images?", precision=0,
            info=f"Between '1' and '{max_num_imgs}' images. Press "\
                    "enter/return to register")
    with gr.Row():  # Dynamic row (images in columns)
        imgs = [gr.Image(label=f"Image {i+1}", visible=True) \
                for i in range(int(nimg_s.value))] + \
                [gr.Image(visible=False) \
                for _ in range(max_num_imgs - int(nimg_s.value))]
        for i, img in enumerate(imgs):  # Set image as "input"
            img.change(lambda _: None, img)
    with gr.Row():  # Dynamic row of output (cluster) images
        imgs2 = [gr.Image(label=f"VLAD Clusters {i+1}", 
                visible=False) for i in range(max_num_imgs)]
    nimg_s.submit(var_num_img, nimg_s, imgs)
    blend_alpha = gr.Number(0.4, label="Blending alpha",
        info="Weight for cluster centers (between 0 and 1). "\
            "Higher (close to 1) means greater emphasis on cluster "\
                "visibility. Lower (closer to 0) will show the "\
                "underlying image more. "\
            "Press enter/return to register")
    bttn1 = gr.Button("Click Me!")  # Cluster assignment
    gr.Markdown("### Status strings")
    out_msg1 = gr.Markdown("Select domain and upload images")
    out_msg2 = gr.Markdown("For descriptor extraction")
    out_msg3 = gr.Markdown("Followed by VLAD assignment")
    out_msg4 = gr.Markdown("Followed by cluster images")
    
    # ---- Utility functions ----
    # A wrapper to batch the images
    def batch_images(data):
        sv = int(data[nimg_s])
        images: List[np.ndarray] = [data[imgs[k]] \
                for k in range(sv)]
        return images
    # A wrapper to unbatch images (and pad to max)
    def unbatch_images(imgs_batch, nimg):
        ret = [gr.Image.update(visible=False) \
                for _ in range(max_num_imgs)]
        if imgs_batch is None or len(imgs_batch) == 0:
            return ret
        for i in range(nimg):   # nimg only to match input layout
            if i < len(imgs_batch):
                img_np = np.array(imgs_batch[i])
            else:
                img_np = None
            ret[i] = gr.Image.update(img_np, visible=True)
        return ret
    
    # ---- Examples ----
    # Two images from each domain
    gr.Examples(
        [
        ["Aerial", 2, 
            "ex_aerial_nardo-air_db-42.png",
            "ex_aerial_nardo-air_qu-42.png",],
        ["Indoor", 2,
            "ex_indoor_17places_db-75.jpg",
            "ex_indoor_17places_qu-75.jpg"],
        ["Urban", 2,
            "ex_urban_oxford_db-75.png",
            "ex_urban_oxford_qu-75.png"],],
        [domain, nimg_s, *imgs],
    )
    
    # ---- Main pipeline ----
    # Get the VLAD cluster assignment images on click
    bttn1.click(get_vlad_clusters, domain, [out_msg1, vlad])\
        .then(batch_images, {nimg_s, *imgs, imgs_batch}, imgs_batch)\
        .then(get_descs, imgs_batch, [patch_descs, out_msg2])\
        .then(assign_vlad, [patch_descs, vlad], 
                [desc_assignments, out_msg3])\
        .then(get_ca_images, 
                [desc_assignments, patch_descs, blend_alpha],
                [imgs_batch, out_msg4])\
        .then(unbatch_images, [imgs_batch, nimg_s], imgs2)
    # If the blending changes now, update the cluster images only
    blend_alpha.submit(get_ca_images, 
            [desc_assignments, patch_descs, blend_alpha],
            [imgs_batch, out_msg4])\
        .then(unbatch_images, [imgs_batch, nimg_s], imgs2)


# Tab for GeM t-SNE projection plot
def tab_gem_tsne():
    d_vals = [k.title() for k in DOMAINS]
    dms = gr.CheckboxGroup(d_vals, value=d_vals, label="Domains",
            info="The domains to use for the t-SNE projection")
    nimg_s = gr.Number(2, label="How many images?", precision=0,
            info=f"Between '1' and '{max_num_imgs}' images. Press "\
                    "enter/return to register")
    with gr.Row():  # Dynamic row (images in columns)
        imgs = [gr.Image(label=f"Image {i+1}", visible=True) \
                for i in range(int(nimg_s.value))] + \
                [gr.Image(visible=False) \
                for _ in range(max_num_imgs - int(nimg_s.value))]
        for i, img in enumerate(imgs):  # Set image as "input"
            img.change(lambda _: None, img)
    nimg_s.submit(var_num_img, nimg_s, imgs)
    tsne_plot = gr.Plot(None, label="tSNE Plot")
    out_msg1 = gr.Markdown("Select domains")
    out_msg2 = gr.Markdown("Upload images")
    out_msg3 = gr.Markdown("Wait for tSNE plots")
    
    # A wrapper to batch the images
    def batch_images(data):
        sv = int(data[nimg_s])
        # images: List[np.ndarray] = [data[imgs[k]] \
        #         for k in range(sv)]
        images: List[np.ndarray] = []
        for k in range(sv):
            img = data[imgs[k]]
            if img is None:
                return None, f"Image {k+1} is None!"
            images.append(img)
        return images, "Images batched"
    
    bttn1 = gr.Button("Click Me!")
    
    # ---- Examples ----
    gr.Examples(
        [
            ["./ex_dining_room.jpeg", "./ex_city_road.jpeg"],
            ["./ex_manhattan_aerial.jpeg", "./ex_city_road.jpeg"],
            ["./ex_dining_room.jpeg", "./ex_manhattan_aerial.jpeg"],
        ],
        [*imgs],
    )
    
    # ---- Main pipeline ----
    # Get the tSNE plot
    bttn1.click(get_gem_descs_cache, dms, [out_msg1, gem_descs])\
        .then(batch_images, {nimg_s, *imgs, imgs_batch}, 
                [imgs_batch, out_msg2])\
        .then(get_add_gem_descs, [imgs_batch, gem_descs],
                [gem_descs, out_msg2])\
        .then(get_tsne_fm_gem, gem_descs, [tsne_pts, out_msg3])\
        .then(plot_tsne, tsne_pts, [tsne_plot, out_msg3])


# Build the interface
with gr.Blocks() as demo:
    # Main header
    gr.Markdown(header_markdown)
    
    # ---- Helper functions ----
    # Variable number of input images (show/hide UI image array)
    def var_num_img(s):
        n = int(s)  # Slider (string) value as int
        assert 1 <= n <= max_num_imgs, f"Invalid num of images: {n}!"
        return [gr.Image.update(label=f"Image {i+1}", visible=True) \
                for i in range(n)] \
            + [gr.Image.update(visible=False) \
                for _ in range(max_num_imgs - n)]
    
    # ---- State declarations ----
    vlad = gr.State()   # VLAD object
    desc_assignments = gr.State()   # Cluster assignments
    imgs_batch = gr.State() # Images as batch
    patch_descs = gr.State()    # Patch descriptors
    gem_descs = gr.State()  # GeM descriptors (of each state)
    tsne_pts = gr.State()   # tSNE points
    
    # ---- All UI elements ----
    with gr.Tab("GeM t-SNE Projection"):
        gr.Markdown(
            """
            ## GeM t-SNE Projection
            
            Select the domains (toggle visibility) for t-SNE plot. \
            Enter the number of images to upload and upload images. \
            Then click the button to get the t-SNE plot.
            
            You can also directly click on one of the examples (at \
            the bottom) to load the data and then click the button \
            to get the t-SNE plot.
            
            The examples have the following images
            - [Manhattan aerial view](https://www.crushpixel.com/stock-photo/aerial-view-midtown-manhattan-849717.html)
            - [Dining room](https://homesfeed.com/formal-dining-room-sets-for-8/)
            - [City road](https://pxhere.com/en/photo/824211)
            
            """)
        tab_gem_tsne()
    
    with gr.Tab("Cluster Visualization"):
        gr.Markdown(
            """
            ## Cluster Visualizations
            
            Select the domain for the images (all should be from the \
            same domain). Enter the number of images to upload. \
            Upload the images. Then click the button to get the \
            cluster assignment images.
            
            You can also directly click on one of the examples (at \
            the bottom) to load the data and then click the button \
            to get the cluster assignment images.
            
            - The `aerial` example is from the Tartan Air dataset
            - The `indoor` example is from the 17Places dataset
            - The `urban` example is from the Oxford dataset
            
            """)
        tab_cluster_viz()

print("Interface build completed")


# %%
# Deploy application
demo.queue().launch(share=share)
print("Application deployment ended, exiting...")
