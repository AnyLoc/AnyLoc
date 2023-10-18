# Apply global (max or average) pooling over Dino V2 descriptors
"""
    Extract Dino v2 features from a facet of a layer and apply
    global pooling (max or average) on them to get global features
    of an image (for VPR).
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
from torchvision import transforms as T
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
from utilities import DinoV2ExtractFeatures
from custom_datasets.baidu_dataloader import Baidu_Dataset
from custom_datasets.oxford_dataloader import Oxford
from custom_datasets.gardens import Gardens


# %%
@dataclass
class LocalArgs:
    # Program arguments (dataset directories and wandb)
    prog: ProgArgs = ProgArgs(wandb_proj="Dino-v2-Descs", 
            wandb_group="GP-Descs")
    # BaseDataset arguments
    bd_args: BaseDatasetArgs = base_dataset_args
    # Experiment identifier (None = don't use)
    exp_id: Union[str, None] = None
    # Pooling type
    pool_method: Literal["average", "max"] = "average"
    # Dino parameters
    # Model type
    model_type: Literal["dinov2_vits14", "dinov2_vitb14", 
            "dinov2_vitl14", "dinov2_vitg14"] = "dinov2_vitg14"
    """
        Model for Dino-v2 to use as the base model.
    """
    # Layer for extracting Dino feature (descriptors)
    desc_layer: int = 31
    # Facet for extracting descriptors
    desc_facet: Literal["query", "key", "value", "token"] = "token"
    # Dataset split for VPR (BaseDataset)
    data_split: Literal["train", "test", "val"] = "test"
    # Sub-sample query images (RAM or VRAM constraints) (1 = off)
    sub_sample_qu: int = 1
    # Sub-sample database images (RAM or VRAM constraints) (1 = off)
    sub_sample_db: int = 1
    # Values for top-k (for monitoring)
    top_k_vals: List[int] = field(default_factory=lambda:\
                                list(range(1, 21, 1)))
    # Show a matplotlib plot for recalls
    show_plot: bool = False


# %%
# ---------------- Functions ----------------
@torch.no_grad()
def build_gp_descs(largs: LocalArgs, vpr_ds: BaseDataset, 
            verbose: bool=True) \
            -> Tuple[torch.Tensor, torch.Tensor]:
    """
        Build globally pooled vectors for database and query images.
        Pooling method taken from `largs.pool_method`
        
        Parameters:
        - largs: LocalArgs  Local arguments for the file
        - vpr_ds: BaseDataset   The dataset containing database and 
                                query images
        - verbose: bool     Prints progress if True
        
        Returns:
        - db_gpds:      Pooled descriptors of database of shape 
                        [n_db, d_dim]
                        - n_db: Number of database images
                        - d_dim: Descriptor dimensionality for the
                            (patch) features
        - qu_gpds:      Pooled descriptors of queries of shape 
                        [n_qu, d_dim], 'n_qu' is num. of queries
    """
    
    # Load Dino feature extractor model
    dino = DinoV2ExtractFeatures(largs.model_type, largs.desc_layer,
                largs.desc_facet, device=device)
    if verbose:
        print("Dino model loaded")
    
    def extract_patch_descriptors(indices):
        patch_descs = []
        for i in tqdm(indices, disable=not verbose):
            img = vpr_ds[i][0].to(device)
            c, h, w = img.shape
            h_new, w_new = (h // 14) * 14, (w // 14) * 14
            img_in = T.CenterCrop((h_new, w_new))(img)[None, ...]
            ret = dino(img_in)
            if largs.pool_method == "average":
                ret = torch.mean(ret, dim=1)
            elif largs.pool_method == "max":
                ret = torch.max(ret, dim=1)[0]
            else:
                raise NotImplementedError(f"ID: {largs.pool_method}")
            patch_descs.append(ret.cpu())
        patch_descs = torch.cat(patch_descs, dim=0) # [N, n_p, d_dim]
        return patch_descs
    
    # Get the database descriptors
    num_db = vpr_ds.database_num
    ds_len = len(vpr_ds)
    assert ds_len > num_db, "Either no queries or length mismatch"
    
    # Get pooled descriptors for the database
    if verbose:
        print("Building pooled descriptors for database...")
    db_indices = np.arange(0, num_db, largs.sub_sample_db)
    # All database descs (local descriptors): [n_db, n_d, d_dim]
    db_gpds = extract_patch_descriptors(db_indices)
    if verbose:
        print(f"Pooled database descriptor shape: {db_gpds.shape}")
    
    # Get pooled descriptors of the queries
    if verbose:
        print("Building pooled descriptors for queries...")
    qu_indices = np.arange(num_db, ds_len, largs.sub_sample_qu)
    # Get global descriptors for queries
    qu_gpds = extract_patch_descriptors(qu_indices)
    if verbose:
        print(f"Full query descriptor shape: {qu_gpds.shape}")
    
    # Return VLADs
    return db_gpds, qu_gpds


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
    elif ds_name=="Oxford_25m":
        vpr_ds = Oxford(ds_dir, override_dist=25)
    elif ds_name=="gardens":
        vpr_ds = Gardens(largs.bd_args,ds_dir,ds_name,largs.data_split)
    else:
        vpr_ds = BaseDataset(largs.bd_args, ds_dir, ds_name, 
                        largs.data_split)
    
    db_gpds, qu_gpds = build_gp_descs(largs, vpr_ds)
    print("--------- Generated pooled descriptors ---------")
    
    print("----- Calculating recalls through top-k matching -----")
    dists, indices, recalls = get_top_k_recall(largs.top_k_vals, 
        db_gpds, qu_gpds, vpr_ds.soft_positives_per_query, 
        sub_sample_db=largs.sub_sample_db, 
        sub_sample_qu=largs.sub_sample_qu)
    print("------------ Recalls calculated ------------")
    
    print("--------------------- Results ---------------------")
    ts = time.strftime(f"%Y_%m_%d_%H_%M_%S")
    caching_directory = largs.prog.cache_dir
    results = {
        "Model-Type": str(largs.model_type),
        "Desc-Layer": str(largs.desc_layer),
        "Desc-Facet": str(largs.desc_facet),
        "Desc-Dim": str(db_gpds.shape[1]),
        "Experiment-ID": str(largs.exp_id),
        "DB-Name": str(ds_name),
        "Num-DB": str(len(db_gpds)),
        "Num-QU": str(len(qu_gpds)),
        "Agg-Method": "GAP" if largs.pool_method == "average" \
                else "GMP",
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
