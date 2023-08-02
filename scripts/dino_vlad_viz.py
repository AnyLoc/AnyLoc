# Doing VLAD with Dino descriptors
"""
    Basic idea is to extract descriptors from [1] and do VLAD on them.
    
    Note: '--prog.wandb-save-qual' is not used
    
    [1]: https://github.com/ShirAmir/dino-vit-features/blob/main/extractor.py#L15
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
from torch.nn import functional as F
from dino_extractor import ViTExtractor
from PIL import Image
import numpy as np
import tyro
from dataclasses import dataclass, field
from utilities import VLAD, get_top_k_recall, seed_everything
import einops as ein
import wandb
import matplotlib.pyplot as plt
import time
import joblib
import traceback
from tqdm.auto import tqdm
from dvgl_benchmark.datasets_ws import BaseDataset
from configs import ProgArgs, prog_args, BaseDatasetArgs, \
        base_dataset_args, device
from typing import Union, Literal, Tuple, List
from custom_datasets.baidu_dataloader import Baidu_Dataset
from custom_datasets.oxford_dataloader import Oxford
from custom_datasets.gardens import Gardens
from custom_datasets.aerial_dataloader import Aerial
from custom_datasets.hawkins_dataloader import Hawkins

import pdb
import cv2
import matplotlib.cm as cm
from skimage import color
import torchvision.transforms as T
from matplotlib.lines import Line2D
import imageio.v2 as imageio


def color_map_color(value, cmap_name='jet', vmin=0, vmax=1):
    norm = plt.Normalize(vmin, vmax)
    cmap = cm.get_cmap(cmap_name)
    rgb = cmap(norm(abs(value)))[:3]  # will return rgba, we take only first 3 so we get rgb
    return rgb


# %%
@dataclass
class LocalArgs:
    # Program arguments (dataset directories and wandb)
    prog: ProgArgs = ProgArgs(wandb_proj="Dino-Descs", 
        wandb_group="viz_vlad_clusters", use_wandb=False)
    # BaseDataset arguments
    bd_args: BaseDatasetArgs = base_dataset_args
    # Experiment identifier (None = don't use)
    exp_id: Union[str, None] = None
    # Dino parameters
    model_type: Literal["dino_vits8", "dino_vits16", "dino_vitb8", 
            "dino_vitb16", "vit_small_patch8_224", 
            "vit_small_patch16_224", "vit_base_patch8_224", 
            "vit_base_patch16_224"] = "dino_vits8"
    """
        Model for Dino to use as the base model.
    """
    # Number of clusters for VLAD
    num_clusters: int = 16
    # Stride for ViT (extractor)
    vit_stride: int = 4
    # Down-scaling H, W resolution for images (before giving to Dino)
    down_scale_res: Tuple[int, int] = (224, 298)
    # Layers for extracting Dino feature (descriptors)
    desc_layers: List[int] = field(default_factory=lambda: [11])
    # Facet for extracting descriptors
    desc_facet: Literal["key", "query", "value", "token"] = "key"
    # Apply log binning to the descriptor
    desc_bin: bool = False
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
def build_vlads(largs: LocalArgs, vpr_ds: BaseDataset, 
            verbose: bool=True) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
        WARNING: docstring is outdated
        
        Build VLAD vectors for database and query images.
        
        Parameters:
        - largs: LocalArgs  Local arguments for the file
        - vpr_ds: BaseDataset   The dataset containing database and 
                                query images
        - verbose: bool     Prints progress if True
        
        Returns:
        - db_vlads:     VLAD descriptors of database of shape 
                        [n_db, vlad_dim]
                        - n_db: Number of database images
                        - vlad_dim: num_clusters * d_dim
                            - d_dim: Descriptor dimensionality
                            - num_clusters: Number of clusters
        - qu_vlads:     VLAD descriptors of queries of shape 
                        [n_qu, vlad_dim], 'n_qu' is num. of queries
    """
    print(largs.model_type,largs.vit_stride,device)
    cache_dir = None
    if largs.cache_vlad_descs:
        cache_dir = f"{largs.prog.cache_dir}/vlad_descs/Dino/" \
                    f"{largs.prog.vg_dataset_name}/" \
                    f"{largs.model_type}-{largs.desc_facet}"
        for l in largs.desc_layers:
            cache_dir += f"L{l:02d}"
        cache_dir += f"-C{largs.num_clusters}"
        print(f"Using cache directory: {cache_dir}")
    
    vlad = VLAD(largs.num_clusters, None, 
            vlad_mode=largs.vlad_assignment, 
            soft_temp=largs.vlad_soft_temp, cache_dir=cache_dir)
    extractor = ViTExtractor(largs.model_type, largs.vit_stride, 
                device=device)
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
        full_db_vlad = []
        # Get global database descriptors
        for i in tqdm(db_indices, disable=(not verbose)):
            img = vpr_ds[i][0]
            img = ein.rearrange(img, "c h w -> 1 c h w").to(device)
            img = F.interpolate(img, largs.down_scale_res)
            descs = []
            for l in largs.desc_layers:
                # Descriptors: [1, num_descs, d_dim]
                desc = extractor.extract_descriptors(img,
                        layer=l, facet=largs.desc_facet,
                        bin=largs.desc_bin)[0]
                descs.append(desc)
            # [n_layers, num_descs, d_dim] for each element
            descs = torch.concat(descs, dim=0)
            descs = ein.rearrange(descs, "l n d -> n (l d)")
            descs = F.normalize(descs, dim=1)
            full_db_vlad.append(descs.cpu())
        full_db_vlad = torch.stack(full_db_vlad)
        if verbose:
            print(f"Database (for VLAD) shape: {full_db_vlad.shape}")
        d_dim = full_db_vlad.shape[2]
        if verbose:
            print(f"Descriptor dimensionality: {d_dim}")
        vlad.fit(ein.rearrange(full_db_vlad, "n k d -> (n k) d"))
        del full_db_vlad
    # VLAD cluster centers loaded
    if verbose:
        print(f"VLAD cluster centers shape: "\
                f"{vlad.c_centers.shape}, ({vlad.c_centers.dtype})")
    
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
        # All database descs (local descriptors): [n_db, n_d, d_dim]
        full_db = []
        # Get global database descriptors
        for i in tqdm(db_indices, disable=(not verbose)):
            img = vpr_ds[i][0]
            img = ein.rearrange(img, "c h w -> 1 c h w").to(device)
            img = F.interpolate(img, largs.down_scale_res)
            descs = []
            for l in largs.desc_layers:
                # Descriptors: [1, num_descs, d_dim]
                desc = extractor.extract_descriptors(img,
                        layer=l, facet=largs.desc_facet,
                        bin=largs.desc_bin)[0]
                descs.append(desc)
            # [n_layers, num_descs, d_dim] for each element
            descs = torch.concat(descs, dim=0)
            descs = ein.rearrange(descs, "l n d -> n (l d)")
            descs = F.normalize(descs, dim=1)
            full_db.append(descs.cpu())
        full_db = torch.stack(full_db)
        if verbose:
            print(f"Full database descriptor shape: {full_db.shape}")
        db_vlads: torch.Tensor = vlad.generate_multi(full_db, 
                db_img_names)
        del full_db
    if verbose:
        print(f"Database VLADs shape: {db_vlads.shape}")
    
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
        full_qu = []
        # Get global descriptors for queries
        for i in tqdm(qu_indices, disable=(not verbose)):
            img = vpr_ds[i][0]
            img = ein.rearrange(img, "c h w -> 1 c h w").to(device)
            img = F.interpolate(img, largs.down_scale_res)
            descs = []
            for l in largs.desc_layers:
                # Descriptors: [1, num_descs, d_dim]
                desc = extractor.extract_descriptors(img,
                        layer=l, facet=largs.desc_facet,
                        bin=largs.desc_bin)[0]
                descs.append(desc)
            # [n_layers, num_descs, d_dim] for each element
            descs = torch.concat(descs, dim=0)
            descs = ein.rearrange(descs, "l n d -> n (l d)")
            descs = F.normalize(descs, dim=1)
            full_qu.append(descs.cpu())
        full_qu = torch.stack(full_qu)
        qu_residuals: torch.Tensor = vlad.generate_multi_res_vec(
                full_qu, qu_img_names)
        if verbose:
            print(f"Full query descriptor shape: {full_qu.shape}")
        del full_qu
    
    if verbose:
        print(f"Query Residuals shape: {qu_residuals.shape}")
    # Return VLADs
    return db_vlads, vlad.c_centers, qu_residuals


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
    elif ds_name=="test_40_midref_rot0" or ds_name=="test_40_midref_rot90":
        vpr_ds = Aerial(largs.bd_args,ds_dir,ds_name,largs.data_split)
    elif ds_name=="hawkins":
        vpr_ds = Hawkins(largs.bd_args,ds_dir,ds_name,largs.data_split)
    else:
        vpr_ds = BaseDataset(largs.bd_args, ds_dir, ds_name, 
                        largs.data_split)
    # Get VLADs of the database 
    #   From: qu_residuals: [n_qu, 4015=(55*73), n_c, d_dim]
    db_vlads, vlad_cluster_centers, qu_residuals = build_vlads(largs, 
                                                    vpr_ds)
    print("--------- Generated VLADs ---------")
    
    print("---------- Visualizing Cluster Centers Assignment ---------")
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
    
    # print(f"Colors: {colors}")
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
        # img_orig = cv2.resize(img_orig,(73,55))
        if largs.qu_indices is None:
            qi_ds = vpr_ds.database_num + i*largs.sub_sample_qu
        else:
            if largs.qu_in_db:
                qi_ds = largs.qu_indices[i]
            else:
                qi_ds = vpr_ds.database_num + largs.qu_indices[i]
        img_orig = vpr_ds[qi_ds][0]#.detach().cpu().numpy()
        resize_T = T.Resize((55,73))    # Dino patches (s=4, ps=8)
        img_orig = resize_T(img_orig)
        # ImageNet normalization
        mu = torch.Tensor([0.485, 0.456, 0.406]).unsqueeze(-1)\
            .unsqueeze(-1)
        std= torch.Tensor([0.229, 0.224, 0.225]).unsqueeze(-1)\
            .unsqueeze(-1)
        # print(img_orig.shape,std.shape,mu.shape)
        img_orig = img_orig * std + mu
        img_orig = img_orig.detach().cpu().numpy()
        img_orig = np.moveaxis(img_orig, 0, -1)*255   # Channels last
        # img_orig = np.asarray(img_orig,dtype=np.uint8)*255
        img_desc = []
        # Loop through all the patches inside image (Dino patches)
        for j in range(qu_residuals.shape[1]):
            cur_res_vec = torch.abs(qu_residuals[i][j])
            res_idx = torch.argmin(torch.sum(cur_res_vec, dim=1))
            img_desc.append(res_idx)
        img_desc = np.reshape(np.asarray(img_desc),(55,73))
        # Color based on the closest clusters
        all_color_img = np.zeros_like(img_orig) # [55, 73, 3]
        for c in range(largs.num_clusters):
            img_idx = (np.argwhere(img_desc==c))
            all_color_img[img_idx[:,0],img_idx[:,1],:] = colors[c]*255
        # Merge cluster color map with original image
        img_original = vpr_ds[qi_ds][0]
        sz = tuple(img_original.shape[1:]) # [H, W]
        resize_T = T.Resize(sz)    # Original shape ;)
        img_original = resize_T(img_original)
        img_original = img_original * std + mu
        img_original = img_original.detach().cpu().numpy()
        img_original = np.moveaxis(img_original, 0, -1)*255
        all_color_img_resized = cv2.resize(all_color_img, sz[::-1], 
                interpolation=cv2.INTER_NEAREST)
        # color_layer_img = cv2.addWeighted(img_orig, 0.7, 
        #         all_color_img, 0.3, 0)
        color_layer_img = cv2.addWeighted(img_original, 0.7,
                all_color_img_resized, 0.3, 0)
        # color_layer_img = cv2.resize(color_layer_img,(298,224))
        color_layer_img = color_layer_img/255   # [0 - 1] float range
        # color_layer_img = np.asarray(color_layer_img,dtype=np.uint8)
        # Plot and save
        fig,ax = plt.subplots()
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
        # print(f"{save_fldr}/{i}.png")
        fig.savefig(f"{save_fldr}/{i}.png")
        plt.close()
    
    # If saving GIF
    frames = []
    if largs.save_gif:
        for i in range(qu_residuals.shape[0]):
            frames.append(imageio.imread(f"{save_fldr}/{i}.png"))
        ts = time.strftime(f"%Y_%m_%d_%H_%M_%S")
        imageio.mimsave(f"{save_fldr}/{ts}.gif", frames, fps=2)
    
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
