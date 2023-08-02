# Creating a VLAD-style descriptor using LSeg
"""
    Uses the cached LSeg descriptors to create VLAD descriptors and
    performs top-k retrieval. Yields the recall values at different
    thresholds.
    1. Use the cache of database descriptors (dense) to create cluster
        centers. This is the vocabulary.
    2. For all images in the cache, create global descriptors using
        the VLAD method (concatenation of normalized residual from the
        cluster centers). Split this for the database and query.
    3. For each query, get the top-k retrieval for the database.
    
    The cache file names have to be in the format that `datasets_vg`
    names files. This is to determine the ground truth (based on 
    distance).
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
import numpy as np
import torch
import time
from dvgl_benchmark.datasets_ws import BaseDataset
import einops as ein
import tyro
import matplotlib.pyplot as plt
from natsort import natsorted
from dataclasses import dataclass, field
from glob import glob
from typing import Literal, Union, List
from tqdm.auto import tqdm
import traceback
import pdb
import wandb
import joblib
# Library modules
from utilities import VLAD, get_top_k_recall, seed_everything
from configs import ProgArgs, prog_args, BaseDatasetArgs, base_dataset_args
from custom_datasets.baidu_dataloader import Baidu_Dataset
from custom_datasets.oxford_dataloader import Oxford
from custom_datasets.gardens import Gardens


# %%
@dataclass
class LocalArgs:
    # Program arguments
    prog: ProgArgs = ProgArgs(wandb_proj="Lseg-vlad", 
        wandb_group="direct_lseg_vpr", vg_dataset_name="pitts30k")
    """
        Program arguments. Used mainly for WandB. The argument 
        `vg_dataset_name` and others are only used for getting ground
        truth (positives for retrievals).
        After changing `vg_dataset_name`, also change the following:
        - `query_cache_dir`: Query LSeg cache directory
        - `db_cache_dir`: Database LSeg cache directory
    """
    # Base dataset arguments (only to get positives/ground truth)
    bd_args: BaseDatasetArgs = base_dataset_args
    # Experiment identifier (None = don't use)
    exp_id: Union[str, None] = None
    # Directory where query LSeg cache is stored
    query_cache_dir: Path = "/scratch/avneesh.mishra/lseg/"\
            "datasets_vg_cache/pitts30k/test/queries"
    # Directory where database LSeg cache is stored
    db_cache_dir: Path = "/scratch/avneesh.mishra/lseg/"\
            "datasets_vg_cache/pitts30k/test/database"
    # Distance threshold for positive retrieval (in m)
    dist_positive_retr: float = 25
    # Database type (DEPRECATED)
    db_type: Literal["vpr_bench", "vg_bench"] = "vg_bench"
    # Dataset split (for dataloader only, not LSeg cache)
    data_split: Literal["train", "test", "val"] = "test"
    # Ground truth (.npy) file (DEPRECATED) (for 'vpr_bench' type)
    gt_npy_file: Union[None, Path] = None
    # Number of clusters (vocabulary size)
    num_clusters: int = 64
    # Top-k values (all values for retrieval)
    top_k_vals: List[int] = field(default_factory=lambda:\
                                list(range(1, 21, 1)))
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
        
        Note that this is nested for the `sub_sample_db` value. This
        means that if `sub_sample_db` is 2 and `sub_sample_db_vlad`
        is 4, then the VLAD clustering will be done on 1/8th of the
        database images (not 1/4th).
    """
    # Sub-sample the database and query image pixels
    sub_sample_pixels: int = 1
    """
        Sub-sample the pixels of the images. This is used to reduce
        the memory usage. This is done after the LSeg descriptors
        are loaded into the memory (first thing).
    """
    # Show the recall plot in the end of program
    show_plot: bool = False
    # Use the intra-cluster normalization
    use_inorm: bool = True
    # Use hard or soft descriptor assignment for VLAD
    vlad_assignment: Literal["hard", "soft"] = "hard"
    # Softmax temperature for VLAD (soft assignment only)
    vlad_soft_temp: float = 1.0


# %%
# Main function
def main(largs: LocalArgs):
    print(f"Arguments: {largs}")
    seed_everything()
    
    # Launch wandb
    if largs.prog.use_wandb:
        wandb_run = wandb.init(project=largs.prog.wandb_proj, 
                entity=largs.prog.wandb_entity, config=largs,
                group=largs.prog.wandb_group, 
                name=largs.prog.wandb_run_name)
    
    # Load data (from cache)
    assert os.path.isdir(largs.query_cache_dir) and \
        os.path.isdir(largs.db_cache_dir)
    _ex = lambda x: os.path.realpath(os.path.expanduser(x))
    query_dir = _ex(largs.query_cache_dir)
    db_dir = _ex(largs.db_cache_dir)
    db_cache_files = natsorted(glob(f"{db_dir}/*.npy"))
    query_cache_files = natsorted(glob(f"{query_dir}/*.npy"))
    # If using sub-samplig
    if largs.sub_sample_db is not None:
        db_cache_files = db_cache_files[::largs.sub_sample_db]
    if largs.sub_sample_qu is not None:
        query_cache_files = query_cache_files[::largs.sub_sample_qu]
    db_cache_files = natsorted(db_cache_files)
    query_cache_files = natsorted(query_cache_files)
    print(f"Found {len(query_cache_files)} query and "\
        f"{len(db_cache_files)} database cache items")
    
    print("----------- Loading descriptors -----------")
    _start = time.time()
    db_descs = torch.stack([torch.from_numpy(np.load(f))[
            ::largs.sub_sample_pixels, ::largs.sub_sample_pixels, :]\
                            .to(torch.float32) \
            for f in tqdm(db_cache_files)]) # [N, H, W, D]
    qu_descs = torch.stack([torch.from_numpy(np.load(f))[
            ::largs.sub_sample_pixels, ::largs.sub_sample_pixels, :] \
                            .to(torch.float32) \
            for f in tqdm(query_cache_files)])
    print(f"Loading took: {time.time()-_start:.3f} seconds")
    print("----------- Loaded descriptors -----------")
    print("----------- Loading Ground Truth -----------")
    datasets_dir = largs.prog.data_vg_dir
    dataset_name = largs.prog.vg_dataset_name
    print(f"Dataset directory: {datasets_dir}")
    print(f"Dataset name: {dataset_name}")
    print(f"Dataset split: {largs.data_split}")
    # Get ground truth for retrievals (through BaseDataset)
    # _, _, soft_pos = BaseDataset.generate_positives_and_utms(
    #     largs.db_type, db_cache_files, query_cache_files, 
    #     largs.gt_npy_file, largs.dist_positive_retr)
    if dataset_name=="baidu_datasets":
        vpr_ds = Baidu_Dataset(largs.bd_args, datasets_dir, dataset_name, 
                            largs.data_split)
    elif dataset_name=="Oxford":
        vpr_ds = Oxford(datasets_dir)
    elif dataset_name=="gardens":
        vpr_ds = Gardens(largs.bd_args, datasets_dir,dataset_name,largs.data_split)
    else:   # `vgl_dataset` or `vpr_bench` dataset
        vpr_ds = BaseDataset(largs.bd_args, datasets_dir, dataset_name, 
                        largs.data_split)
    soft_pos = vpr_ds.get_positives()
    print("Loaded ground truth for positives")
    print("----------- Creating VLAD descriptors -----------")
    _start = time.time()
    vlad = VLAD(largs.num_clusters, intra_norm=largs.use_inorm, 
                vlad_mode=largs.vlad_assignment, 
                soft_temp=largs.vlad_soft_temp)
    db_d = ein.rearrange(db_descs, "n h w d -> (n h w) d")\
            [::largs.sub_sample_db_vlad, ...]
    vlad.fit(db_d)  # Database descriptors for VLAD cluster centers
    db_descs_i = ein.rearrange(db_descs, "n h w d -> n (h w) d")
    db_vlads = vlad.generate_multi(db_descs_i)
    print(f"VLAD 'db' took: {time.time()-_start:.3f} seconds")
    _start = time.time()
    qu_descs_i = ein.rearrange(qu_descs, "n h w d -> n (h w) d")
    qu_vlads = vlad.generate_multi(qu_descs_i)
    print(f"VLAD 'queries' took: {time.time()-_start:.3f} seconds")
    print("----------- Created VLAD descriptors -----------")
    
    print("----------- Getting top-k recalls -----------")
    # pdb.set_trace()
    dists, indices, recalls = get_top_k_recall(largs.top_k_vals, 
        db_vlads, qu_vlads, soft_pos)
    
    print("--------------------- Results ---------------------")
    ts = time.strftime(f"%Y_%m_%d_%H_%M_%S")
    caching_directory = largs.prog.cache_dir
    results = {
        "Timestamp": str(ts),
        "Exp-ID": str(largs.exp_id),
        "Database-Dir": str(largs.db_cache_dir),
        "Query-Dir": str(largs.query_cache_dir),
        "Num-Clusters": str(largs.num_clusters),
        "DB-Name": str(largs.prog.vg_dataset_name),
        "Num-DB": str(len(db_cache_files)),
        "Num-QU": str(len(query_cache_files)),
        "Timestamp": str(ts)
    }
    print("Results:")
    for k in results:
        print(f"- {k}: {results[k]}")
    print("- Recalls: ")
    for k in recalls.keys():
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
            plt.title(f"{plt_title} - {wandb_run.name}")
        else:
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


if __name__ == "__main__" and ("ipykernel" not in sys.argv[0]):
    largs = tyro.cli(LocalArgs)
    _start = time.time()
    try:
        main(largs)
    except Exception as exc:
        print(f"Exception: {exc}")
        traceback.print_exc()
    except:
        print("Error occurred")
        traceback.print_exc()
    finally:
        print(f"Program ended in {time.time()-_start:.3f} seconds")
        exit(0)


# %%
# Experiments

