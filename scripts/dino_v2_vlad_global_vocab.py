# Doing VLAD (global vocabulary) with Dino V2 descriptors
"""
    DEPRECATED: Use dino_v2_global_vocab_vlad.py instead
    
    Basic idea is to explore the layers and facets of Dino-v2 and do
    VLAD over them. The global vocabulary is created by using 
    descriptors from all datasets.
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
from custom_datasets.aerial_dataloader import Aerial
from custom_datasets.hawkins_dataloader import Hawkins
from custom_datasets.vpair_dataloader import VPAir
from custom_datasets.laurel_dataloader import Laurel
from custom_datasets.eiffel_dataloader import Eiffel
from custom_datasets.global_dataloader import Global_Dataloader


# %%
@dataclass
class LocalArgs:
    # Program arguments (dataset directories and wandb)
    prog: ProgArgs = ProgArgs(wandb_proj="Dino-v2-Descs", 
        wandb_group="VLAD-Descs")
    # BaseDataset arguments
    bd_args: BaseDatasetArgs = base_dataset_args
    # Experiment identifier (None = don't use)
    exp_id: Union[str, None] = None
    # Dino parameters
    # Model type
    model_type: Literal["dinov2_vits14", "dinov2_vitb14", 
            "dinov2_vitl14", "dinov2_vitg14"] = "dinov2_vits14"
    """
        Model for Dino-v2 to use as the base model.
    """
    # Number of clusters for VLAD
    num_clusters: int = 8
    # Layer for extracting Dino feature (descriptors)
    desc_layer: int = 11
    # Facet for extracting descriptors
    desc_facet: Literal["query", "key", "value", "token"] = "key"
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
# ---------------- Functions ----------------
@torch.no_grad()
def build_vlads(largs: LocalArgs, vpr_ds: BaseDataset,vpr_global_ds: BaseDataset,
            verbose: bool=True) \
            -> Tuple[torch.Tensor, torch.Tensor]:
    """
        Build VLAD vectors for database and query images.
        
        Parameters:
        - largs: LocalArgs  Local arguments for the file
        - vpr_ds: BaseDataset   The dataset containing database and 
                                query images
        - vpr_global_ds: BaseDataset   The dataset containing database images for multiple runs
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
            patch_descs.append(ret.cpu())
        patch_descs = torch.cat(patch_descs, dim=0) # [N, n_p, d_dim]
        return patch_descs

    def extract_patch_descriptors_global(indices):
        patch_descs = []
        for i in tqdm(indices, disable=not verbose):
            img = vpr_global_ds[i][0]
            c, h, w = img.shape
            h_new, w_new = (h // 14) * 14, (w // 14) * 14
            img_in = T.CenterCrop((h_new, w_new))(img)[None, ...].to(device)
            ret = dino(img_in)
            patch_descs.append(ret.cpu())
        patch_descs = torch.cat(patch_descs, dim=0) # [N, n_p, d_dim]
        return patch_descs

    # Get the database descriptors
    num_db = vpr_ds.database_num
    ds_len = len(vpr_ds)
    assert ds_len > num_db, "Either no queries or length mismatch"
    
    #Get the global database descriptors
    global_ds_len = len(vpr_global_ds)
    print("Length of global database is: ", global_ds_len)

    if vlad.can_use_cache_vlad():
        if verbose:
            print("Valid cache found, using it")
        vlad.fit(None)  # Nothing to fit (restore cache)
    else:
        # Get cluster centers in the VLAD
        if verbose:
            print("Building VLAD cluster centers...")
        db_indices_global = np.arange(0, global_ds_len, largs.sub_sample_db_vlad)
        # Database descriptors (for VLAD clusters): [n_db, n_d, d_dim]
        full_db_vlad = extract_patch_descriptors_global(db_indices_global)
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
        full_qu = []
        # Get global descriptors for queries
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
    elif ds_name=="test_40_midref_rot90" or ds_name=="test_40_midref_rot0" or ds_name=="GNSS_Tartan":
        vpr_ds = Aerial(largs.bd_args,ds_dir,ds_name,largs.data_split)
    elif ds_name=="hawkins" or ds_name=="hawkins_long_corridor":
        vpr_ds = Hawkins(largs.bd_args,ds_dir,ds_name,largs.data_split)
    elif ds_name=="VPAir":
        vpr_ds = VPAir(largs.bd_args,ds_dir,ds_name,largs.data_split)
    elif ds_name=="laurel_caverns":
        vpr_ds = Laurel(largs.bd_args,ds_dir,ds_name,largs.data_split)
    elif ds_name=="eiffel":
        vpr_ds = Eiffel(largs.bd_args,ds_dir,ds_name,largs.data_split)
    else:
        vpr_ds = BaseDataset(largs.bd_args, ds_dir, ds_name, 
                        largs.data_split)
    
    ds_name_list = ["st_lucia", "pitts30k", "17places","baidu_datasets","Oxford","gardens","hawkins_long_corridor","eiffel","VPAir","GNSS_Tartan","laurel_caverns"]
    vpr_global_ds = Global_Dataloader(largs.bd_args, ds_dir, ds_name_list, 
                        largs.data_split)

    db_vlads, qu_vlads = build_vlads(largs, vpr_ds,vpr_global_ds)
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
        "Desc-Layer": str(largs.desc_layer),
        "Desc-Facet": str(largs.desc_facet),
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


# %%
# Experimental section

# %%
largs = LocalArgs(prog=ProgArgs(vg_dataset_name="gardens"), 
        sub_sample_db=10, sub_sample_qu=10, 
        sub_sample_db_vlad=1, model_type="dinov2_vits14", 
        desc_layer=8, num_clusters=64)
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
elif ds_name=="test_40_midref_rot90" or ds_name=="test_40_midref_rot0":
    vpr_ds = Aerial(largs.bd_args,ds_dir,ds_name,largs.data_split)
elif ds_name=="hawkins" or ds_name=="hawkins_long_corridor":
    vpr_ds = Hawkins(largs.bd_args,ds_dir,ds_name,largs.data_split)
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
print(f"Dino dim: {db_vlads.shape[1]//largs.num_clusters}")
print(f"VLAD dim: {db_vlads.shape[1]}")

# %%
vpr_ds.get_image_relpaths(10)

# %%
