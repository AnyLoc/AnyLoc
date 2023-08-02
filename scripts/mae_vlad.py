# Doing VLAD with descriptors from MAE
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
import joblib
import wandb
import traceback
from configs import ProgArgs, prog_args, BaseDatasetArgs, \
        base_dataset_args, device
from typing import Union, Literal, Tuple, List
from utilities import VLAD, get_top_k_recall, seed_everything
from dvgl_benchmark.datasets_ws import BaseDataset
from custom_datasets.baidu_dataloader import Baidu_Dataset
from custom_datasets.oxford_dataloader import Oxford
from custom_datasets.gardens import Gardens



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
    # If True, use the CLS token in VLAD, else discard it.
    use_cls_token: bool = False
    # Number of clusters for VLAD
    num_clusters: int = 8
    # Image size for the MAE (default models are 224, 224 image size)
    img_size: int = 224
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
    # Show a matplotlib plot for recalls
    show_plot: bool = False
    # Use hard or soft descriptor assignment for VLAD
    vlad_assignment: Literal["hard", "soft"] = "hard"
    # Softmax temperature for VLAD (soft assignment only)
    vlad_soft_temp: float = 1.0
    # Caching configuration
    cache_vlad_descs: bool = False


# %%
@torch.no_grad()
def build_vlads(largs: LocalArgs, vpr_ds: BaseDataset, 
            verbose: bool=True) \
            -> Tuple[torch.Tensor, torch.Tensor]:
    """
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
    ims = largs.img_size    # Square input image size
    
    def extract_patch_descriptors(indices):
        patch_descs = []    # Patch descriptors for each image
        for i in tqdm(indices, disable=(not verbose)):
            # img = vpr_ds[i][0][[2, 1, 0], :, :]
            img = vpr_ds[i][0]
            c, h, w = img.shape
            h_n, w_n = map(lambda x: int((x//ims)*ims), [h, w])
            img = tvf.center_crop(img[None, ...], (h_n, w_n))
            img_patches = ein.rearrange(img, 
                    "b c (nh h) (nw w) -> (nh nw) b c h w", b=1,
                    nh=h_n//ims, nw=w_n//ims)
            latents_patches = []    # [n_si, n_patch, d_dim]
            for sub_img in img_patches:
                _, _, _, latents = model(sub_img.to(device), 
                        mask_ratio=0)
                if not largs.use_cls_token:
                    latents = latents[:, 1:, :]
                latents_patches.append(latents)
            latents_patches = torch.cat(latents_patches, dim=0)
            latents_patches = latents_patches.detach().cpu()
            latents_patches = ein.rearrange(latents_patches,
                    "n_si n_patch d_dim -> (n_si n_patch) d_dim")
            patch_descs.append(latents_patches)
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
        full_db = extract_patch_descriptors(db_indices)
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
    qu_indices = np.arange(num_db, ds_len, largs.sub_sample_qu)
    qu_img_names = vpr_ds.get_image_relpaths(qu_indices)
    if vlad.can_use_cache_ids(qu_img_names):
        if verbose:
            print("Valid cache found, using it")
        qu_vlads = vlad.generate_multi([None] * len(qu_indices), 
                qu_img_names)
    else:
        if verbose:
            print("Valid cache not found, doing forward pass")
        full_qu = extract_patch_descriptors(qu_indices)
        if verbose:
            print(f"Full query descriptor shape: {full_qu.shape}")
        qu_vlads: torch.Tensor = vlad.generate_multi(full_qu,
                qu_img_names)
        del full_qu
    if verbose:
        print(f"Query VLADs shape: {qu_vlads.shape}")
    
    # Return VLADs
    return db_vlads, qu_vlads


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
    
    db_vlads, qu_vlads = build_vlads(largs, vpr_ds)
    print("--------- Generated VLADs ---------")
    
    print("----- Calculating recalls through top-k matching -----")
    dists, indices, recalls = get_top_k_recall(largs.top_k_vals, 
        db_vlads, qu_vlads, vpr_ds.soft_positives_per_query, 
        sub_sample_db=largs.sub_sample_db, 
        sub_sample_qu=largs.sub_sample_qu)
    print("------------ Recalls calculated ------------")
    
    print("--------------------- Results ---------------------")
    ts = time.strftime(f"%Y_%m_%d_%H_%M_%S")
    caching_directory = largs.prog.cache_dir
    results = {
        "Model-Type": str(largs.mae_model),
        "Checkpoint": str(os.path.basename(largs.ckpt_path)),
        "Desc-Dim": str(db_vlads.shape[1]//largs.num_clusters),
        "VLAD-Dim": str(db_vlads.shape[1]),
        "Num-Clusters": str(largs.num_clusters),
        "Experiment-ID": str(largs.exp_id),
        "DB-Name": str(ds_name),
        "Num-DB": str(len(db_vlads)),
        "Num-QU": str(len(qu_vlads)),
        "Timestamp": str(ts)
    }
    print("Results: ")
    for k in results:
        print(f"- {k}: {results[k]}")
    print("- Recalls: ")
    for k in recalls:
        results[f"R@{k}"] = recalls[k]
        print(f"  - R@{k}: {recalls[k]:.5f}")
    if largs.show_plot:
        plt.plot(recalls.keys(), recalls.values())
        plt.ylim(0, 1)
        plt.xticks(largs.top_k_vals)
        plt.xlabel("top-k values")
        plt.ylabel(r"% recall")
        plt_title = "Recall curve"
        if largs.exp_id is not None:
            plt_title = f"{plt_title} - Exp {largs.exp_id}"
        if largs.prog.use_wandb:
            plt_title = f"{plt_title} - {wandb_run.name}"
        plt.title(plt_title)
        plt.show()
    
    # Log to WandB
    if largs.prog.use_wandb:
        wandb.log(results)
        for tk in recalls:
            wandb.log({"Recall-All": recalls[tk]}, step=int(tk))
    
    # Add retrievals
    results["Qual-Dists"] = dists
    results["Qual-Indices"] = indices
    save_res_file = None
    if largs.exp_id == True:
        save_res_file = caching_directory
    elif type(largs.exp_id) == str:
        save_res_file = f"{caching_directory}/experiments/"\
                        f"{largs.exp_id}"
    if save_res_file is not None:
        if not os.path.isdir(save_res_file):
            os.makedirs(save_res_file)
        save_res_file = f"{save_res_file}/results_{ts}.gz"
        print(f"Saving result in: {save_res_file}")
        joblib.dump(results, save_res_file)
    else:
        print("Not saving results")
    
    if largs.prog.use_wandb:
        wandb.finish()
    print("--------------------- END ---------------------")
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
# Experiments


# %%
largs = LocalArgs(prog=ProgArgs(vg_dataset_name="Oxford"), 
        sub_sample_db=5, sub_sample_qu=5, 
        sub_sample_db_vlad=2, ckpt_path="./../models/mae/"\
            "mae_visualize_vit_large.pth", num_clusters=32)
print(f"Arguments: {largs}")

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
_start = time.time()
db_vlads, qu_vlads = build_vlads(largs, vpr_ds)
print(f"Building VLAD took {time.time()-_start:.3f} seconds")

# %%
dists, indices, recalls = get_top_k_recall(largs.top_k_vals, 
    db_vlads, qu_vlads, vpr_ds.soft_positives_per_query, 
    sub_sample_db=largs.sub_sample_db, 
    sub_sample_qu=largs.sub_sample_qu)

# %%
plt.plot(recalls.keys(), recalls.values())
plt.ylim(0, 1)
plt.xticks(largs.top_k_vals)
plt.xlabel("top-k values")
plt.ylabel(r"% recall")
plt_title = "Recall curve"
if largs.exp_id is not None:
    plt_title = f"{plt_title} - Exp {largs.exp_id}"
plt.title(plt_title)
plt.show()

# %%
print(f"Descriptor dim: {db_vlads.shape[1]//largs.num_clusters}")
print(f"VLAD dim: {db_vlads.shape[1]}")

# %%
