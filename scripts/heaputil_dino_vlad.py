# Using HeapUtil visualizations for Dino
"""
    HeapUtil from [1] is a tool for visualizing VLAD clusters.
    
    [1]: https://github.com/Nik-V9/HEAPUtil
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


# %%
@dataclass
class LocalArgs:
    # Program arguments (dataset directories and wandb)
    prog: ProgArgs = ProgArgs(wandb_proj="Dino-Descs", 
        wandb_group="Direct-Descs")
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
    num_clusters: int = 8
    # Stride for ViT (extractor)
    vit_stride: int = 4
    # Down-scaling H, W resolution for images (before giving to Dino)
    down_scale_res: Tuple[int, int] = (224, 298)
    # Layer for extracting Dino feature (descriptors)
    desc_layer: int = 11
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
    # Values for top-k (for monitoring)
    top_k_vals: List[int] = field(default_factory=lambda:\
                                list(range(1, 21, 1)))
    # Use hard or soft descriptor assignment for VLAD
    vlad_assignment: Literal["hard", "soft"] = "hard"
    # Softmax temperature for VLAD (soft assignment only)
    vlad_soft_temp: float = 1.0
    # Override the query indices (None = don't use override)
    qu_indices: Union[List[int], None] = None
    """
        Override the query indices. This is useful for debugging or
        for running experiments on a subset of the dataset (for 
        visualization purposes).
    """
    # Caching configuration
    cache_vlad_descs: bool = True


# %%
@torch.no_grad()
def build_vlads(largs: LocalArgs, vpr_ds: BaseDataset, 
            verbose: bool=True) \
            -> Tuple[torch.Tensor, torch.Tensor]:
    """
        Build VLAD descriptors for database and query images.
        
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
    cache_dir = None
    if largs.cache_vlad_descs:
        cache_dir = f"{largs.prog.cache_dir}/vlad_descs/Dino/" \
                    f"{largs.prog.vg_dataset_name}/" \
                    f"{largs.model_type}-{largs.desc_facet}-" \
                    f"L{largs.desc_layer}-C{largs.num_clusters}"
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
            desc = extractor.extract_descriptors(img,
                    layer=largs.desc_layer, facet=largs.desc_facet,
                    bin=largs.desc_bin) # [1, 1, num_descs, d_dim]
            full_db_vlad.append(desc.squeeze().cpu())
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
    
    if verbose:
        print("Building VLAD of queries...")
    if largs.qu_indices is None:
        qu_indices = np.arange(num_db, ds_len, largs.sub_sample_qu)
    else:
        qu_indices = np.array(largs.qu_indices) + num_db
    qu_img_names = vpr_ds.get_image_relpaths(qu_indices)
    if vlad.can_use_cache_ids(qu_img_names):
        if verbose:
            print("Valid cache found, using it")
        qu_vlads = vlad.generate_multi([None] * len(qu_indices), 
                qu_img_names)
    else:
        if verbose:
            print("Valid cache not found, doing forward pass")
        full_qu = []
        # Get global descriptors for queries
        for i in tqdm(qu_indices, disable=(not verbose)):
            img = vpr_ds[i][0]
            img = ein.rearrange(img, "c h w -> 1 c h w").to(device)
            img = F.interpolate(img, largs.down_scale_res)
            desc = extractor.extract_descriptors(img, 
                    layer=largs.desc_layer, facet=largs.desc_facet, 
                    bin=largs.desc_bin) # [1, 1, num_descs, d_dim]
            full_qu.append(desc.squeeze().cpu())
        full_qu = torch.stack(full_qu)
        if verbose:
            print(f"Full query descriptor shape: {full_qu.shape}")
        qu_vlads: torch.Tensor = vlad.generate_multi(full_qu,
                qu_img_names)
        del full_qu
    if verbose:
        print(f"Query VLADs shape: {qu_vlads.shape}")
    
    if verbose:
        print("Building cluster assignments...")
    if vlad.can_use_cache_ids(qu_img_names, only_residuals=True):
        if verbose:
            print("Residuals cache found, using it")
        qu_residuals: torch.Tensor = vlad.generate_multi_res_vec(
                [None] * len(qu_indices), qu_img_names)
    else:
        full_qu = []
        # Get global descriptors for queries
        for i in tqdm(qu_indices, disable=(not verbose)):
            img = vpr_ds[i][0]
            img = ein.rearrange(img, "c h w -> 1 c h w").to(device)
            img = F.interpolate(img, largs.down_scale_res)
            desc = extractor.extract_descriptors(img, 
                    layer=largs.desc_layer, facet=largs.desc_facet, 
                    bin=largs.desc_bin) # [1, 1, num_descs, d_dim]
            full_qu.append(desc.squeeze().cpu())
        full_qu = torch.stack(full_qu)
        qu_residuals: torch.Tensor = vlad.generate_multi_res_vec(
                full_qu, qu_img_names)
        if verbose:
            print(f"Full query descriptor shape: {full_qu.shape}")
        del full_qu
    if verbose: # [n_qu, n_patches = 4015 = (55*73), n_cluster, d_dim]
        print(f"Query residuals shape: {qu_residuals.shape}")
    # Loop through all queries
    img_descs = []
    for i in tqdm(range(qu_residuals.shape[0])):
        img_desc = []
        # Loop through all the patches inside image (Dino patches)
        for j in range(qu_residuals.shape[1]):
            cur_res_vec = torch.abs(qu_residuals[i][j])
            # Cluster with smallest distance
            res_idx = torch.argmin(torch.sum(cur_res_vec, dim=1))
            img_desc.append(res_idx)
        img_desc = torch.tensor(img_desc).view((55, 73))
        img_descs.append(img_desc)
    img_descs = torch.stack(img_descs)  # All query clusters
    if verbose:
        print(f"Query cluster assignments shape: {img_descs.shape}")
    
    return qu_indices, qu_vlads, img_descs


# %%
@torch.no_grad()
def main(largs: LocalArgs):
    pass


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
largs = LocalArgs(ProgArgs(vg_dataset_name="pitts30k"), 
        qu_indices=list(range(10)))


# %%
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
print(f"Dataset Class: {vpr_ds.__class__.__name__}")
gt_pos_qu = vpr_ds.get_positives()
print(f"Ground truth positives class: {type(gt_pos_qu)}")
if type(gt_pos_qu) is list:
    print("Converting ground truth positives to numpy array")
    gt_pos_qu = np.array([np.array(i) for i in gt_pos_qu], 
                        dtype=object)
gt_neg_qu = np.array([
    np.delete(np.arange(vpr_ds.database_num), i) for i in gt_pos_qu
], dtype=object)


# %%
qu_inds, qu_vlads, qu_cluster_assignments = build_vlads(largs, vpr_ds, True)

# %%
qu_cl_res = F.interpolate(qu_cluster_assignments[None, ...]\
        .to(torch.float32), (480, 640))[0].to(qu_cluster_assignments)

# %%


# %%
