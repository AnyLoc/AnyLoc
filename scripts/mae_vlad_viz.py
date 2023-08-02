# Visualizing VLAD clusters for MAE
"""
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
from torch import nn
from torch.nn import functional as F
from torchvision.transforms import functional as tvf
import numpy as np
import tyro
import einops as ein
from tqdm.auto import tqdm
import models_mae
from dataclasses import dataclass, field
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import joblib
import wandb
import traceback
import cv2 as cv
import imageio.v2 as imageio
from configs import ProgArgs, prog_args, BaseDatasetArgs, \
        base_dataset_args, device
from typing import Union, Literal, Tuple, List
from utilities import VLAD, get_top_k_recall, seed_everything
from dvgl_benchmark.datasets_ws import BaseDataset
from custom_datasets.baidu_dataloader import Baidu_Dataset
from custom_datasets.oxford_dataloader import Oxford
from custom_datasets.gardens import Gardens


def color_map_color(value, cmap_name='jet', vmin=0, vmax=1):
    norm = plt.Normalize(vmin, vmax)
    cmap = cm.get_cmap(cmap_name)
    rgb = cmap(norm(abs(value)))[:3]  # will return rgba, we take only first 3 so we get rgb
    return rgb


# %%
@dataclass
class LocalArgs:
    # Program arguments (dataset directories and wandb)
    prog: ProgArgs = ProgArgs(use_wandb=False, 
            vg_dataset_name="17places")
    # BaseDataset arguments
    bd_args: BaseDatasetArgs = base_dataset_args
    # Experiment identifier (None = don't use)
    exp_id: Union[str, None] = None
    # MAE model parameters
    ckpt_path: Path = "./models/mae/"\
            "mae_visualize_vit_large_ganloss.pth"
    """
        Path to the MAE model checkpoint.
    """
    # MAE model type (should be compatible with the checkpoint)
    mae_model: Literal["mae_vit_base_patch16", 
            "mae_vit_large_patch16", "mae_vit_huge_patch14"] = \
                "mae_vit_large_patch16"
    # Mask ratio for MAE (0.0 = no masking) (should be in [0, 1])
    mask_ratio: float = 0.0
    # If True, use the CLS token in VLAD, else discard it.
    use_cls_token: bool = False
    # Center crop image
    center_crop: bool = False
    """
        The MAE takes a square image as input. If this is True, the
        image is center-cropped to a square image before being reduced
        to the required size. If False, the image is directly resized
        to the required size ('down_scale_res').
    """
    # Number of clusters for VLAD
    num_clusters: int = 16
    # Down-scaling H, W resolution for images (before giving to MAE)
    down_scale_res: Tuple[int, int] = (224, 224)
    # Dataset split for VPR (BaseDataset)
    data_split: Literal["train", "test", "val"] = "test"
    # Sub-sample query images (RAM or VRAM constraints) (1 = off)
    sub_sample_qu: int = 1
    # Sub-sample database images (RAM or VRAM constraints) (1 = off)
    sub_sample_db: int = 1
    # Sub-sample database images for VLAD clustering only
    sub_sample_db_vlad: int = 1
    """
        Use sub-sampling for creating the VLAD cluster centers. Use
        this to reduce the RAM usage during the clustering process.
        Unlike `sub_sample_qu` and `sub_sample_db`, this is only used
        for clustering and not for the actual VLAD computation.
    """
    # Override the query indices (None = don't use override)
    qu_indices: Union[List[int], None] = None
    # Override queries to be placed in database images
    qu_in_db: bool = False
    """
        If True, the 'qu_indices' are treated in the database set and
        not the query set. This setting is valid only if 'qu_indices'
        is not None.
    """
    # Values for top-k (for monitoring)
    top_k_vals: List[int] = field(default_factory=lambda:\
                                list(range(1, 21, 1)))
    # Show a matplotlib plot for recalls
    show_plot: bool = False
    # Use hard or soft descriptor assignment for VLAD
    vlad_assignment: Literal["hard", "soft"] = "hard"
    # Softmax temperature for VLAD (soft assignment only)
    vlad_soft_temp: float = 1.0
    # Caching configuration
    cache_vlad_descs: bool = False
    # Save the resultant images as a GIF (filename is timestamp)
    save_gif: bool = False


# %%
# ---------------- Functions ----------------
@torch.no_grad()
def build_res(largs: LocalArgs, vpr_ds: BaseDataset, 
            verbose: bool=True, run_db: bool=True) \
            -> Tuple[Union[torch.Tensor, None], torch.Tensor, 
                        torch.Tensor]:
    """
        Builds and returns the cluster residuals for queries. Also
        has the option to return the database VLADs as well.
        
        Parameters:
        - largs: LocalArgs  Local arguments for the file
        - vpr_ds: BaseDataset   The dataset containing database and 
                                query images
        - verbose: bool     Prints progress if True
        - run_db: bool      If True, also returns the database VLADs.
                            Else, returns None for the database VLADs.
        
        Returns:
        - 
    """
    cache_dir = None
    if largs.cache_vlad_descs:
        cache_dir = f"{largs.prog.cache_dir}/vlad_descs/MAE/" \
                    f"{largs.prog.vg_dataset_name}/" \
                    f"{largs.mae_model}-C{largs.num_clusters}"
        if verbose:
            print(f"Using cache directory: {cache_dir}")
    # Build VLAD representations (global descriptors)
    vlad = VLAD(largs.num_clusters, None, 
            vlad_mode=largs.vlad_assignment,
            soft_temp=largs.vlad_soft_temp, cache_dir=cache_dir)
    # Load MAE model
    ckpt_path = os.path.realpath(os.path.expanduser(largs.ckpt_path))
    assert os.path.isfile(ckpt_path), \
            f"Checkpoint not found: {ckpt_path}"
    model: models_mae.MaskedAutoencoderViT = getattr(models_mae, 
        largs.mae_model)(ret_latents=True)
    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu")
    msg = model.load_state_dict(ckpt["model"], strict=False)
    if verbose:
        print(f"Model loaded: {msg}")
    model = model.to(device)
    if verbose:
        print(f"Model moved to: {device}")
    
    def extract_patch_descriptors(indices):
        patch_descs = []
        for i in tqdm(indices, disable=(not verbose)):
            img = vpr_ds[i][0]
            img = ein.rearrange(img, "c h w -> 1 c h w").to(device)
            if largs.center_crop:
                img = tvf.center_crop(img, min(img.shape[2:]))
            img = F.interpolate(img, largs.down_scale_res)
            loss, y, mask, latent = model(img, 
                    mask_ratio=largs.mask_ratio)
            if not largs.use_cls_token: # [1, n_p=(n_h*n_w), d_dim]
                latent: torch.Tensor = latent[:, 1:, :]
            patch_descs.append(latent.squeeze().cpu())
        patch_descs = torch.stack(patch_descs)
        return patch_descs
    
    # Get the database descriptors
    num_db = vpr_ds.database_num
    ds_len = len(vpr_ds)
    assert ds_len > num_db, "Either no queries or length mismatch"
    if vlad.can_use_cache_vlad():
        if verbose:
            print("Valid cache found, using it")
        vlad.fit(None)  # Nothing to fit (restore cache)
    else:
        # Get cluster centers in the VLAD
        if verbose:
            print("Building VLAD cluster centers...")
        db_indices = np.arange(0, num_db, largs.sub_sample_db_vlad)
        # Database descriptors (for VLAD clusters): [n_db, n_d, d_dim]
        full_db_vlad = extract_patch_descriptors(db_indices)
        if verbose:
            print(f"Database (for VLAD) shape: {full_db_vlad.shape}")
        d_dim = full_db_vlad.shape[2]   # Should be 1024
        if verbose:
            print(f"Descriptor dimensionality: {d_dim}")
        vlad.fit(ein.rearrange(full_db_vlad, "n k d -> (n k) d"))
        del full_db_vlad
    if verbose:
        print(f"VLAD cluster centers shape: "\
            f"{vlad.c_centers.shape}, ({vlad.c_centers.dtype})")
    
    if run_db:
        # Get VLADs of the database
        if verbose:
            print("Building VLADs for database...")
        db_indices = np.arange(0, num_db, largs.sub_sample_db)
        db_img_names = vpr_ds.get_image_relpaths(db_indices)
        if vlad.can_use_cache_ids(db_img_names):
            if verbose:
                print("Valid cache found, using it")
            db_vlads = vlad.generate_multi([None] * len(db_indices), 
                    db_img_names)
        else:
            if verbose:
                print("Valid cache not found, doing forward pass")
            # All database descs (local): [n_db, n_d, d_dim]
            full_db = extract_patch_descriptors(db_indices)
            if verbose:
                print(f"Full database descriptor shape: " \
                        f"{full_db.shape}")
            db_vlads: torch.Tensor = vlad.generate_multi(full_db, 
                    db_img_names)
            del full_db
        if verbose:
            print(f"Database VLADs shape: {db_vlads.shape}")
    else:
        db_vlads = None
    
    # Get VLADs of the queries
    if verbose:
        print("Building VLADs for queries...")
    if largs.qu_indices is None:
        qu_indices = np.arange(num_db, ds_len, largs.sub_sample_qu)
    else:
        if largs.qu_in_db:
            qu_indices = np.array(largs.qu_indices)
        else:
            qu_indices = np.array(largs.qu_indices) + num_db
    qu_img_names = vpr_ds.get_image_relpaths(qu_indices)
    if vlad.can_use_cache_ids(qu_img_names, only_residuals=True):
        if verbose:
            print("Valid cache found, using it")
        qu_residuals: torch.Tensor = vlad.generate_multi_res_vec(
                [None] * len(qu_indices), qu_img_names)
    else:
        if verbose:
            print("Valid cache not found, doing forward pass")
        full_qu = extract_patch_descriptors(qu_indices)
        if verbose:
            print(f"Full query descriptor shape: {full_qu.shape}")
        qu_residuals: torch.Tensor = vlad.generate_multi_res_vec(
                full_qu, qu_img_names)
        if verbose:
            print(f"Full query descriptor shape: {full_qu.shape}")
        del full_qu
    
    if verbose:
        print(f"Query Residuals shape: {qu_residuals.shape}")
    # Return VLADs
    return db_vlads, vlad.c_centers, qu_residuals


# %%
@torch.no_grad()
def main(largs: LocalArgs):
    print(f"Arguments: {largs}")
    seed_everything(42)
    
    if largs.prog.use_wandb:
        # Launch WandB
        wandb_run = wandb.init(project=largs.prog.wandb_proj, 
                entity=largs.prog.wandb_entity, config=largs,
                group=largs.prog.wandb_group, 
                name=largs.prog.wandb_run_name)
        print(f"Initialized WandB run: {wandb_run.name}")
    
    print("--------- Generating VLADs ---------")
    ds_dir = largs.prog.data_vg_dir
    ds_name = largs.prog.vg_dataset_name
    print(f"Dataset directory: {ds_dir}")
    print(f"Dataset name: {ds_name}, split: {largs.data_split}")
    # Load dataset
    if ds_name=="baidu_datasets":
        vpr_ds = Baidu_Dataset(largs.bd_args, ds_dir, ds_name, 
                            largs.data_split)
    elif ds_name=="Oxford":
        vpr_ds = Oxford(ds_dir)
    elif ds_name=="gardens":
        vpr_ds = Gardens(largs.bd_args,ds_dir,ds_name,largs.data_split)
    else:
        vpr_ds = BaseDataset(largs.bd_args, ds_dir, ds_name, 
                        largs.data_split)
    # Get VLADs of the database 
    #   From: qu_residuals: [n_qu, 196=(14*14), n_c, d_dim]
    db_vlads, vlad_cluster_centers, qu_residuals = build_res(largs, 
                                                    vpr_ds)
    print("--------- Generated VLADs ---------")
    
    print("-------- Visualizing Cluster Centers Assignment -------")
    # Visualize VLAD clusters
    colors = np.zeros((largs.num_clusters,3))
    legend_lines = []
    legend_nums = []
    for j in range(largs.num_clusters):
        colors[j,:] = color_map_color(j/(largs.num_clusters-1))
        custom_line = Line2D([0], [0], color = color_map_color(
                j/(largs.num_clusters-1)), lw=4)
        legend_lines.append(custom_line)
        legend_nums.append(str(j))
    # Ensure that save directory exists
    save_fldr = largs.prog.cache_dir
    if largs.exp_id is not None:
        save_fldr = f"{save_fldr}/experiments/{largs.exp_id}"
    save_fldr = f"{save_fldr}/vlad_clusters_viz"
    save_fldr = os.path.realpath(os.path.expanduser(save_fldr))
    if not os.path.exists(save_fldr):
        os.makedirs(save_fldr)
    else:
        print(f"WARNING: Folder {save_fldr} exists! overwriting...")
    print(f"Saving images in: {save_fldr}")
    
    # Loop through all query images
    for i in tqdm(range(qu_residuals.shape[0])):
        # img_orig = cv2.imread(vpr_ds.q_abs_paths[i])
        # img_orig = cv2.resize(img_orig,(14,14))
        if largs.qu_indices is None:
            qi_ds = vpr_ds.database_num + i*largs.sub_sample_qu
        else:
            if largs.qu_in_db:
                qi_ds = largs.qu_indices[i]
            else:
                qi_ds = vpr_ds.database_num + largs.qu_indices[i]
        img = vpr_ds[qi_ds][0]#.detach().cpu().numpy()
        if largs.center_crop:
            img = tvf.center_crop(img, min(img.shape[1:]))
        img = tvf.resize(img, (14, 14))
        # ImageNet normalization
        mu = torch.Tensor([0.485, 0.456, 0.406]).unsqueeze(-1)\
            .unsqueeze(-1)
        std= torch.Tensor([0.229, 0.224, 0.225]).unsqueeze(-1)\
            .unsqueeze(-1)
        img = img * std + mu
        img_orig = img.detach().cpu().numpy()
        img_orig = ein.rearrange(img_orig, 'c h w -> h w c') * 255
        img_desc = []
        # Loop through all the patches inside image (MAE patches)
        for j in range(qu_residuals.shape[1]):
            cur_res_vec = torch.abs(qu_residuals[i][j])
            res_idx = torch.argmin(torch.sum(cur_res_vec, dim=1))
            img_desc.append(res_idx)
        img_desc = np.reshape(np.asarray(img_desc), (14, 14))
        # Color based on the closest clusters
        all_color_img = np.zeros_like(img_orig)
        for c in range(largs.num_clusters):
            img_idx = (np.argwhere(img_desc==c))
            all_color_img[img_idx[:,0],img_idx[:,1],:] = colors[c]*255
        # Merge cluster color map with original image
        img_original = vpr_ds[qi_ds][0]
        if largs.center_crop:
            img_original = tvf.center_crop(img_original, 
                    min(img_original.shape[1:]))
        img_original = img_original * std + mu
        img_original = img_original.detach().cpu().numpy()
        sz = tuple(img_original.shape[1:]) # [H, W]
        img_original = ein.rearrange(img_original, 'c h w -> h w c')
        img_original = img_original * 255
        all_color_img_resized = cv.resize(all_color_img, sz[::-1], 
                interpolation=cv.INTER_NEAREST)
        color_layer_img = cv.addWeighted(img_original, 0.7,
                all_color_img_resized, 0.3, 0)
        color_layer_img = color_layer_img/255
        fig, ax = plt.subplots()
        splits = vpr_ds.get_image_paths()[qi_ds].split("/")[-2:]
        f_title = str(os.path.join(*splits))
        ax.set_title(f_title)
        im = ax.imshow(color_layer_img)
        if largs.num_clusters <= 16:
            ax.legend(legend_lines, legend_nums, loc='upper left', 
                    bbox_to_anchor=(1.025, 1.05))
        else:
            # cb = plt.colorbar(im)
            # TODO: Want a custom colorbar with color labels?
            pass
        # plt.show()
        fig.set_tight_layout(True)
        fig.savefig(f"{save_fldr}/{i}.png")
        plt.close()
    
    # If saving GIF
    frames = []
    if largs.save_gif:
        for i in range(qu_residuals.shape[0]):
            frames.append(imageio.imread(f"{save_fldr}/{i}.png"))
        ts = time.strftime(f"%Y_%m_%d_%H_%M_%S")
        imageio.mimsave(f"{save_fldr}/{ts}.gif", frames, fps=1)
    
    print("----- Finished visualization of cluster centers -----")


# %%
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
# Experiments


# %%
largs = LocalArgs(prog=ProgArgs(vg_dataset_name="17places",
    use_wandb=False), sub_sample_db=5, sub_sample_qu=5, 
    sub_sample_db_vlad=2, ckpt_path="./../models/mae/"\
        "mae_visualize_vit_large_ganloss.pth", save_gif=True, 
    num_clusters=16, center_crop=True)
print(f"Arguments: {largs}")

# %%
_start = time.time()
try:
    main(largs)
except:
    print("Unhandled exception")
    traceback.print_exc()
finally:
    print(f"Program ended in {time.time()-_start:.3f} seconds")


# %%
ds_dir = largs.prog.data_vg_dir
ds_name = largs.prog.vg_dataset_name
print(f"Dataset directory: {ds_dir}")
print(f"Dataset name: {ds_name}, split: {largs.data_split}")
# Load dataset
if ds_name=="baidu_datasets":
    vpr_ds = Baidu_Dataset(largs.bd_args, ds_dir, ds_name, 
                        largs.data_split)
elif ds_name=="Oxford":
    vpr_ds = Oxford(ds_dir)
elif ds_name=="gardens":
    vpr_ds = Gardens(largs.bd_args,ds_dir,ds_name,largs.data_split)
else:
    vpr_ds = BaseDataset(largs.bd_args, ds_dir, ds_name, 
                    largs.data_split)

# %%
img = vpr_ds[10][0]
img_s = (img - img.min())/(img.max() - img.min() + 1e-8)
plt.imshow(img_s.detach().cpu().numpy().transpose(1,2,0))

# %%
img = vpr_ds[10][0]
img = tvf.center_crop(img, min(img.shape[1:]))
img_s = (img - img.min())/(img.max() - img.min() + 1e-8)
plt.imshow(img_s.detach().cpu().numpy().transpose(1,2,0))

# %%
