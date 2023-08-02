# Doing VLAD with SAM descriptors
"""
    Basic idea is to extract descriptors from [1] and do VLAD on them.
    
    Note: '--prog.wandb-save-qual' is not used
    
    [1]: https://github.com/facebookresearch/segment-anything
"""

# %%
import os
import sys
from pathlib import Path
# Set the './../' from the script folder
# Also set the './../segment_anything/' from the script folder
dir_name = None
try:
    dir_name = os.path.dirname(os.path.realpath(__file__))
except NameError:
    print('WARN: __file__ not found, trying local')
    dir_name = os.path.abspath('')
lib_path = os.path.realpath(f'{Path(dir_name).parent}')
seg_path = os.path.join(lib_path,'segment-anything')
# Add to path
if lib_path not in sys.path:
    print(f'Adding library path: {lib_path} to PYTHONPATH')
    sys.path.append(lib_path)
else:
    print(f'Library path {lib_path} already in PYTHONPATH')

if seg_path not in sys.path:
    print(f'Adding library path: {seg_path} to PYTHONPATH')
    sys.path.append(seg_path)
else:
    print(f'Library path {seg_path} already in PYTHONPATH')

# %%
import torch
from torch.nn import functional as F

from segment_anything import build_sam, SamPredictor, sam_model_registry

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
import cv2

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
    # SAM parameters
    model_type: Literal["vit_l","vit_b","vit_h"] = "vit_l"
    """
        Model for SAM to use as the base model.
    """
    # model_path: Literal = "/ocean/projects/cis220039p/jkarhade/data/sam_model/sam_vit_h_4b8939.pth"
    # Number of clusters for VLAD
    num_clusters: int = 8
    # Down-scaling H, W resolution for images (before giving to SAM)
    down_scale_res: Tuple[int, int] = (224, 298)
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
    # ViT output layer number
    out_layer_num: int = -1
    # ViT use neck
    use_neck:bool = True

# %%
# ---------------- Functions ----------------
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

    # SAM extractor
    sam = sam_model_registry[largs.model_type](checkpoint= "/ocean/projects/cis220039p/jkarhade/data/sam_model/sam_vit_b_01ec64.pth")
    sam.to(device)
    predictor = SamPredictor(sam,use_neck=largs.use_neck,out_layer_num=largs.out_layer_num)
    
    # Get the database descriptors
    num_db = vpr_ds.database_num
    ds_len = len(vpr_ds)
    assert ds_len > num_db, "Either no queries or length mismatch"
    
    # Get cluster centers in the VLAD
    if verbose:
        print("Building VLAD cluster centers...")
    db_indices = np.arange(0, num_db, largs.sub_sample_db_vlad)
    # Database descriptors (for VLAD clusters): [n_db, n_d, d_dim]
    full_db_vlad = []
    # Get global database descriptors
    for i in tqdm(db_indices, disable=(not verbose)):
        img = vpr_ds[i][0]
        # img = ein.rearrange(img, "c h w -> 1 c h w")#.to(device)
        # print(img.shape)
        #F.interpolate(img, largs.down_scale_res)
        # img = img.squeeze(0)

        img = cv2.resize(img,largs.down_scale_res,interpolation = cv2.INTER_AREA)
        predictor.set_image(img)
        desc = predictor.get_image_embedding()

        # import pdb; pdb.set_trace()
        desc = desc.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2).unsqueeze(dim=1)
        # print(desc.shape)
        full_db_vlad.append(desc.squeeze().cpu())
    full_db_vlad = torch.stack(full_db_vlad)
    if verbose:
        print(f"Database (for VLAD) shape: {full_db_vlad.shape}")
    d_dim = full_db_vlad.shape[2]
    if verbose:
        print(f"Descriptor dimensionality: {d_dim}")
    vlad = VLAD(largs.num_clusters, d_dim, 
            vlad_mode=largs.vlad_assignment, 
            soft_temp=largs.vlad_soft_temp)
    vlad.fit(ein.rearrange(full_db_vlad, "n k d ->(n k) d"))
    if verbose:
        print(f"VLAD cluster centers shape: {vlad.c_centers.shape}")
    del full_db_vlad
    
    # Get VLADs of the database
    if verbose:
        print("Building VLADs for database...")
    db_indices = np.arange(0, num_db, largs.sub_sample_db)
    # All database descriptors (local descriptors): [n_db, n_d, d_dim]
    full_db = []
    # Get global database descriptors
    for i in tqdm(db_indices, disable=(not verbose)):
        img = vpr_ds[i][0]
        # img = ein.rearrange(img, "c h w -> 1 c h w").to(device)
        #F.interpolate(img, largs.down_scale_res)
        # img = img.squeeze(0)

        img = cv2.resize(img,largs.down_scale_res,interpolation = cv2.INTER_AREA)
        predictor.set_image(img)
        desc = predictor.get_image_embedding()
        desc = desc.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2).unsqueeze(dim=1)

        full_db.append(desc.squeeze().cpu())
    full_db = torch.stack(full_db)
    if verbose:
        print(f"Full database descriptor shape: {full_db.shape}")
    db_vlads: torch.Tensor = vlad.generate_multi(full_db)
    if verbose:
        print(f"Database VLADs shape: {db_vlads.shape}")
    del full_db
    
    # Get VLADs of the queries
    if verbose:
        print("Building VLADs for queries...")
    qu_indices = np.arange(num_db, ds_len, largs.sub_sample_qu)
    full_qu = []
    # Get global descriptors for queries
    for i in tqdm(qu_indices, disable=(not verbose)):
        img = vpr_ds[i][0]
        # img = ein.rearrange(img, "c h w -> 1 c h w").to(device)
        #F.interpolate(img, largs.down_scale_res)
        # img = img.squeeze(0)
        
        img = cv2.resize(img,largs.down_scale_res,interpolation = cv2.INTER_AREA)
        predictor.set_image(img)
        desc = predictor.get_image_embedding()
        desc = desc.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2).unsqueeze(dim=1)
        full_qu.append(desc.squeeze().cpu())
    full_qu = torch.stack(full_qu)
    print(full_qu.shape)
    if verbose:
        print(f"Full query descriptor shape: {full_qu.shape}")
    qu_vlads: torch.Tensor = vlad.generate_multi(full_qu)
    if verbose:
        print(f"Query VLADs shape: {qu_vlads.shape}")
    del full_qu
    
    # Return VLADs
    return db_vlads, qu_vlads

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
                            largs.data_split,use_SAM=True)
    elif ds_name=="Oxford":
        vpr_ds = Oxford(ds_dir,use_SAM=True)
    elif ds_name=="gardens":
        vpr_ds = Gardens(largs.bd_args,ds_dir,ds_name,largs.data_split,use_SAM=True)
    elif ds_name=="test_40_midref_rot0" or ds_name=="test_40_midref_rot90":
        vpr_ds = Aerial(largs.bd_args,ds_dir,ds_name,largs.data_split)
    else:
        vpr_ds = BaseDataset(largs.bd_args, ds_dir, ds_name, 
                        largs.data_split,use_SAM=True)
    
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
        "Model-Type": str(largs.model_type),
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
