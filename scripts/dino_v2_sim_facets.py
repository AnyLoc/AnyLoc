# Visualize similarity across facets for Dino-v2
"""
    Script tries to replicate results like Figure 4 of [1].
    Given an image with pixel coordinates (for similarity check), a
    Dino-v2 model, and a layer, and a target image
    
    - Get the response maps in source and target images (upscaled)
    - Do facet-wise similarity for the selected pixel in source in the
        target image
    
    [1]: Amir, S., Gandelsman, Y., Bagon, S., & Dekel, T. (2021). Deep ViT Features as Dense Visual Descriptors. ArXiv. /abs/2112.05814
"""

# %%
import os
import sys
from pathlib import Path
# Set the './../' from the script folder
dir_name = None
try:
    dir_name = os.path.dirname(os.path.realpath(__file__))
except NameError:
    print('WARN: __file__ not found, trying local')
    dir_name = os.path.abspath('')
lib_path = os.path.realpath(f'{Path(dir_name).parent}')
# Add to path
if lib_path not in sys.path:
    print(f'Adding library path: {lib_path} to PYTHONPATH')
    sys.path.append(lib_path)
else:
    print(f'Library path {lib_path} already in PYTHONPATH')


# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as tvf
import numpy as np
import einops as ein
from scipy.stats import mode
from PIL import Image
import cv2 as cv
import tyro
import time
import traceback
import joblib
from utilities import DinoV2ExtractFeatures, seed_everything
import matplotlib.pyplot as plt
from configs import ProgArgs, prog_args, BaseDatasetArgs, \
        base_dataset_args, device
from typing import Tuple, Dict, Literal, Optional
from dataclasses import dataclass, field
from dvgl_benchmark.datasets_ws import BaseDataset
from custom_datasets.baidu_dataloader import Baidu_Dataset
from custom_datasets.oxford_dataloader import Oxford
from custom_datasets.gardens import Gardens
from custom_datasets.aerial_dataloader import Aerial
from custom_datasets.hawkins_dataloader import Hawkins
from custom_datasets.vpair_dataloader import VPAir
from custom_datasets.laurel_dataloader import Laurel
from custom_datasets.eiffel_dataloader import Eiffel
from custom_datasets.vpair_distractor_dataloader import VPAir_Distractor


# %%
@dataclass
class LocalArgs:
    # Program arguments (dataset directories and wandb)
    prog: ProgArgs = ProgArgs(use_wandb=False)
    # BaseDataset arguments
    bd_args: BaseDatasetArgs = base_dataset_args
    # Dataset split for VPR (BaseDataset)
    data_split: Literal["train", "test", "val"] = "test"
    # Source image index
    src_ind: int = 0
    # Dataset for selecting (indexing) source image
    src_in: Literal["database", "query"] = "database"
    # Target image index
    tgt_ind: int = 0
    # Dataset for selecting (indexing) target image
    tgt_in: Literal["database", "query"] = "query"
    # Pixel location in source image (X = right, Y = down)
    pix_loc: Tuple[int, int] = (555, 200)
    # If True, show matplotlib plots (else don't show)
    show_plts: bool = False
    # Option to resize if images are of different sizes
    assert_sizes: bool = True
    # ----------------- Dino parameters -----------------
    # Model type
    model_type: Literal["dinov2_vits14", "dinov2_vitb14", 
            "dinov2_vitl14", "dinov2_vitg14"] = "dinov2_vits14"
    """
        Model for Dino-v2 to use as the base model.
    """
    # Layer for extracting Dino feature (descriptors)
    desc_layer: int = 11
    # For overriding the size of image (w, h)
    force_size: Optional[Tuple[int, int]] = None


# %%
@torch.no_grad()
def get_sims(simg: np.ndarray, timg: np.ndarray, 
            pix_loc: Tuple[int, int], dino_model: str, 
            dino_layer: int, interp_mode = "nearest", 
            device = "cuda", assert_sizes=True) \
            -> Dict[str, np.ndarray]:
    """
        Get similarity maps for a given pixel location (of the source
        image) in the target image.
        
        - simg: Source image. Shape: [H, W, 3]
        - timg: Target image. Shape: [H, W, 3]
        - pix_loc:  Pixel location in the source image. [H, W] value.
        - dino_model:   Model name for Dino-v2 model
        - dino_layer:   Layer to use for the descriptor extraction
        - device:   Torch device to use for computations (and model)
        - assert_sizes: If True, the sizes are not changed (assert
            error when different). If False and the sizes differ, then
            the target image is resized to the source images shape.
        
        Returns:
        - sim_res:  Dictionary containing similarities for different
                    facets in the Dino model. Facet is the key (str)
                    and value is the similarity map of shape [H, W, 1]
                    (0-1 float values - cosine similarity value)
    """
    sim_res = dict()    # Store results here
    if simg.shape != timg.shape and not assert_sizes:
        timg_r = cv.resize(timg, simg.shape[1::-1],
                interpolation=cv.INTER_NEAREST)
        timg = timg_r
    assert simg.shape == timg.shape, "Images not of same shape"
    h, w, three = simg.shape
    tf = tvf.Compose([  # Transform numpy image to torch image
        tvf.ToTensor(),
        tvf.CenterCrop([(h//14)*14, (w//14)*14]),
        # ImageNet mean and std
        tvf.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
    ])
    simg_pt = tf(simg)[None, ...].to(device)
    timg_pt = tf(timg)[None, ...].to(device)
    for facet in ["key", "query", "token", "value"]:
        # Dino feature extractor
        dino = DinoV2ExtractFeatures(dino_model, dino_layer, facet,
                                    device=device)
        # Extract features
        res_s = dino(simg_pt).detach().cpu()
        res_t = dino(timg_pt).detach().cpu()
        del dino
        # Process image (to original resolutions)
        res_s_img = ein.rearrange(res_s[0], 
                "(p_h p_w) d -> d p_h p_w", 
                p_h=int(simg_pt.shape[-2]/14), 
                p_w=int(simg_pt.shape[-1]/14))[None, ...]
        res_s_img = F.interpolate(res_s_img, mode='nearest',
                size=(simg.shape[0], simg.shape[1]))
        res_t_img = ein.rearrange(res_t[0], 
                "(p_h p_w) d -> d p_h p_w",
                p_h=int(timg_pt.shape[-2]/14),
                p_w=int(timg_pt.shape[-1]/14))[None, ...]
        res_t_img = F.interpolate(res_t_img, mode='nearest',
                size=(timg.shape[0], timg.shape[1]))
        # Extract similarity map
        s_pix = res_s_img[[0], ..., pix_loc[1], pix_loc[0]]
        s_pix = ein.repeat(s_pix, "1 d -> 1 d h w", 
                h=res_s_img.shape[-2], w=res_s_img.shape[-1])
        sim = F.cosine_similarity(res_t_img, s_pix, dim=1)
        sim: np.ndarray = ein.rearrange(sim, "1 h w -> h w 1")\
                .detach().cpu().numpy()
        sim_res[facet] = sim
    return sim_res


# %%
@torch.no_grad()
def main(largs: LocalArgs):
    print(f"Arguments: {largs}")
    seed_everything(42)
    
    ds_dir = largs.prog.data_vg_dir
    ds_name = largs.prog.vg_dataset_name
    print(f"Dataset directory: {ds_dir}")
    print(f"Dataset name: {ds_name}, split: {largs.data_split}")
    bd_args, ds_split = largs.bd_args, largs.data_split
    # Load dataset
    if ds_name=="baidu_datasets":
        vpr_ds = Baidu_Dataset(bd_args, ds_dir, ds_name, ds_split)
    elif ds_name=="Oxford":
        vpr_ds = Oxford(ds_dir)
    elif ds_name=="gardens":
        vpr_ds = Gardens(bd_args, ds_dir, ds_name, ds_split)
    elif ds_name.startswith("Tartan_GNSS"):
        vpr_ds = Aerial(bd_args, ds_dir, ds_name, ds_split)
    elif ds_name.startswith("hawkins"): # Use only long_corridor
        vpr_ds = Hawkins(bd_args, ds_dir,"hawkins_long_corridor", 
                ds_split)
    elif ds_name=="VPAir":
        vpr_ds = VPAir(bd_args, ds_dir, ds_name, ds_split)
        vpr_distractor_ds = VPAir_Distractor(bd_args, ds_dir, 
                ds_name, ds_split)
    elif ds_name=="laurel_caverns":
        vpr_ds = Laurel(bd_args, ds_dir, ds_name, ds_split)
    elif ds_name=="eiffel":
        vpr_ds = Eiffel(bd_args, ds_dir, ds_name, ds_split)
    else:
        vpr_ds = BaseDataset(bd_args, ds_dir, ds_name, ds_split)
    # Load images
    s_ind = largs.src_ind
    if largs.src_in == "query":
        s_ind += vpr_ds.database_num
    t_ind = largs.tgt_ind
    if largs.tgt_in == "query":
        t_ind += vpr_ds.database_num
    src_img = vpr_ds.get_image_paths()[s_ind]
    tgt_img = vpr_ds.get_image_paths()[t_ind]
    pix_loc = largs.pix_loc    # (W, H)
    show_plts = largs.show_plts
    dino_model = largs.model_type
    dino_layer = largs.desc_layer
    dst_dir = f"{largs.prog.cache_dir}/dino_v2_sim_facets/"\
            f"{dino_model}_L{dino_layer}/{ds_name}"
    sl = f"{largs.src_ind}{largs.src_in[0].upper()}"
    tl = f"{largs.tgt_ind}{largs.tgt_in[0].upper()}"
    save_fname = f"I{sl}-{tl}_Px{pix_loc[0]}_Py{pix_loc[1]}"
    simg = Image.open(src_img)
    timg = Image.open(tgt_img)
    if largs.force_size is not None:
        simg = simg.resize(largs.force_size)
        timg = timg.resize(largs.force_size)
    simg_np, timg_np = np.array(simg), np.array(timg)
    if largs.assert_sizes:
        assert simg_np.shape == timg_np.shape, "Shape mismatch"
    if simg_np.shape != timg_np.shape:
        print(f"Warn: Images not of same shape (S: {simg_np.shape},"\
                f" T: {timg_np.shape}). Resizing target to source.")
        timg_np = cv.resize(timg_np, simg_np.shape[1::-1],
                interpolation=cv.INTER_NEAREST)
    h, w, three = simg_np.shape
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)
        print(f"Directory created: {dst_dir}")
    else:
        print(f"Destination directory '{dst_dir}' already exists!")
    # Show image
    if show_plts:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.title("Source Image")
        plt.imshow(simg_np)
        plt.plot(*pix_loc, 'rx')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.title("Target Image")
        plt.imshow(timg_np)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    print("Getting similarities")
    sims = get_sims(simg_np, timg_np, pix_loc, dino_model, dino_layer, 
            device=device, assert_sizes=largs.assert_sizes)
    # Maximum locations in the max (mode = most recurrent)
    print("Getting maximum locations")
    key_max = mode(np.argwhere(sims["key"].max() == sims["key"]), 
            axis=0).mode[0, :2]
    query_max = mode(np.argwhere(sims["query"].max()==sims["query"]),
            axis=0).mode[0, :2]
    value_max = mode(np.argwhere(sims["value"].max()==sims["value"]),
            axis=0).mode[0, :2]
    token_max = mode(np.argwhere(sims["token"].max()==sims["token"]),
            axis=0).mode[0, :2]
    
    print("Saving results")
    nm = lambda x: (((x/2.0) + 0.5) * 255).astype(np.uint8)
    # Marker properties
    mp = {"ms": 20, "mew": 2, "mec": 'white', "alpha": 0.5}
    # Colors for the markers
    # cl = {
    #     "key": "#ffff00",   # Yellow
    #     "query": "#66ff2e", # Green
    #     "value": "#0000ff", # Blue
    #     "token": "#ff0000", # Red
    #     "prompt": "#ff00ff" # Purple
    # }
    cl = {  # We are using this for the paper
        "key": "tab:pink",
        "query": "tab:brown",
        "value": "tab:orange",
        "token": "tab:purple",
        "prompt": "red"
    }
    # Figure
    fig = plt.figure(figsize=(36, 6), dpi=500)
    gs = fig.add_gridspec(1, 6)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(simg_np)
    ax1.plot(*pix_loc, 'o', c=cl["prompt"], **mp)
    ax1.axis('off')
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(timg_np)
    ax2.plot(key_max[1], key_max[0], 'o', label="key", 
             c=cl["key"], **mp)
    ax2.plot(query_max[1], query_max[0], 'o', label="query",
             c=cl["query"], **mp)
    ax2.plot(value_max[1], value_max[0], 'o', label="value",
             c=cl["value"], **mp)
    ax2.plot(token_max[1], token_max[0], 'o', label="token",
             c=cl["token"], **mp)
    ax2.axis('off')
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title("Key")
    ax3.imshow(nm(sims["key"]), vmin=0, vmax=255, cmap="jet")
    ax3.axis('off')
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.set_title("Query")
    ax4.imshow(nm(sims["query"]), vmin=0, vmax=255, cmap="jet")
    ax4.axis('off')
    ax5 = fig.add_subplot(gs[0, 4])
    ax5.set_title("Value")
    ax5.imshow(nm(sims["value"]), vmin=0, vmax=255, cmap="jet")
    ax5.axis('off')
    ax6 = fig.add_subplot(gs[0, 5])
    ax6.set_title("Token")
    ax6.imshow(nm(sims["token"]), vmin=0, vmax=255, cmap="jet")
    ax6.axis('off')
    fig.legend(loc="lower center", ncol=4, 
            bbox_to_anchor=(0.1, 0.125, 0.8, 0.1), mode="expand")
    fig.set_tight_layout(True)
    # Save combined, source, target, key, query, value, and token imgs
    fig.savefig(f"{dst_dir}/{save_fname}.png")  # Main figure
    extent = ax1.get_window_extent().transformed(   # Source
            fig.dpi_scale_trans.inverted())
    fig.savefig(f"{dst_dir}/{save_fname}_source.png", 
            bbox_inches=extent)
    extent = ax2.get_window_extent().transformed(   # Target
            fig.dpi_scale_trans.inverted())
    fig.savefig(f"{dst_dir}/{save_fname}_target.png",
            bbox_inches=extent)
    extent = ax3.get_window_extent().transformed(   # Key
            fig.dpi_scale_trans.inverted())
    fig.savefig(f"{dst_dir}/{save_fname}_key.png",
            bbox_inches=extent)
    extent = ax4.get_window_extent().transformed(   # Query
            fig.dpi_scale_trans.inverted())
    fig.savefig(f"{dst_dir}/{save_fname}_query.png",
            bbox_inches=extent)
    extent = ax5.get_window_extent().transformed(   # Value
            fig.dpi_scale_trans.inverted())
    fig.savefig(f"{dst_dir}/{save_fname}_value.png",
            bbox_inches=extent)
    extent = ax6.get_window_extent().transformed(   # Token
            fig.dpi_scale_trans.inverted())
    fig.savefig(f"{dst_dir}/{save_fname}_token.png",
            bbox_inches=extent)
    # All results as joblib dump
    res = {"source": simg_np, "target": timg_np, "similarities": sims,
            "max": {"key": key_max, "query": query_max, 
                    "value": value_max, "token": token_max},
            "pix_loc": pix_loc}
    joblib.dump(res, f"{dst_dir}/{save_fname}.gz")
    if show_plts:
        fig.show()
    print(f"Saved in file: {dst_dir}/{save_fname}.[png,gz]")


if __name__ == "__main__" and ("ipykernel" not in sys.argv[0]):
    largs = tyro.cli(LocalArgs, description=__doc__)
    _start = time.time()
    try:
        main(largs)
    except:
        print("Unhandled exception")
        traceback.print_exc()
    finally:
        print(f"Program ended in {time.time()-_start:.3f} seconds")
        exit(0)


# %%
