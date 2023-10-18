# PCA downsampling on joint global vocabularies
"""
    This script basically gets recalls from a set of (given) global.
    descriptors.
    
    About the PCA downsampling experiment:
    1. For a domain, generate global descriptors (domain vocabulary
        VLADs of database and queries) and store in `vlad_descs`
        directory. Use `dino_v2_global_vocab_vlad.py` with the
        `save_vlad_descs` argument to save descriptors to a location.
    2. Generate PCA (downsampled) of the global descriptors (of all
        datasets, jointly) through `joing_pca_project.py` and store in
        `pca_<D_out>` directory.
    3. This script: It just takes PCA arguments (only to locate the 
        folder from 2.), dataset name, base directory, and outputs the
        recalls.
    
    You'll still need to have all datasets (as the program asserts for
    shapes).
"""

# %%
import os
import sys
from pathlib import Path
# Set the "./../" from the script folder
dir_name = None
try:
    dir_name = os.path.dirname(os.path.realpath(__file__))
except NameError:
    print("WARNING: __file__ not found, trying local")
    dir_name = os.path.abspath('')
lib_path = os.path.realpath(f"{Path(dir_name).parent}")
# Add to path
if lib_path not in sys.path:
    print(f"Adding library path: {lib_path} to PYTHONPATH")
    sys.path.append(lib_path)
else:
    print(f"Library path {lib_path} already in PYTHONPATH")


# %%
import time
import tyro
import wandb
import torch
import joblib
import traceback
import numpy as np
from matplotlib import pyplot as plt
from typing import Union, Literal, List
from dataclasses import dataclass, field
from torch.nn.functional import normalize
# Internal imports
from utilities import get_top_k_recall, seed_everything
from configs import ProgArgs, BaseDatasetArgs, base_dataset_args
# Only to assert the number of database and queries
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
    # Program arguments (only dataset and WandB)
    prog: ProgArgs = ProgArgs()
    # BaseDataset arguments
    bd_args: BaseDatasetArgs = base_dataset_args
    # Dataset split for VPR (BaseDataset)
    data_split: Literal["train", "test", "val"] = "test"
    # Base directory (where all files from description are stored)
    base_dir: Path = "/scratch/avneesh.mishra/vl-vpr/cache/experiments/pca_downsample"
    # Experiment identifier (None = don't use)
    exp_id: Union[str, None] = None
    # PCA dimensionality reduction (logging and directory ONLY)
    pca_dim_reduce: int = 512
    # PCA whitening (logging ONLY)
    pca_whitening: bool = True
    # Values for top-k (for monitoring)
    top_k_vals: List[int] = field(default_factory=lambda:\
                                list(range(1, 21, 1)))
    # Show a matplotlib plot for recalls
    show_plot: bool = False
    # Dino parameters (don't change if using AnyLoc-DINOv2-VLAD)
    # Model type (logging ONLY)
    model_type: Literal["dinov2_vitg14", "dino_vits8"] = \
            "dinov2_vitg14"
    # Layer for extracting Dino feature (descriptors) (logging ONLY)
    desc_layer: int = 31
    # Facet for extracting descriptors (logging ONLY)
    desc_facet: Literal["query", "key", "value", "token"] = "value"
    # Number of clusters for VLAD (logging ONLY)
    num_clusters: int = 32
    # Sub-sample query images (RAM or VRAM constraints) (1 = off)
    sub_sample_qu: int = 1
    # Sub-sample database images (RAM or VRAM constraints) (1 = off)
    sub_sample_db: int = 1


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
    
    print("---- Reading database and query descriptors ----")
    _ex = lambda x: os.path.realpath(os.path.expanduser(x))
    base_dir = _ex(largs.base_dir)
    pca_dir = f"{base_dir}/pca_{largs.pca_dim_reduce}"
    assert os.path.isdir(pca_dir), f"NotFound: {pca_dir = }"
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
    elif ds_name.startswith("Tartan_GNSS"):
        vpr_ds = Aerial(largs.bd_args,ds_dir,ds_name,largs.data_split)
    elif ds_name.startswith("hawkins"): # Use only long_corridor
        vpr_ds = Hawkins(largs.bd_args,ds_dir,"hawkins_long_corridor",largs.data_split)
    elif ds_name=="VPAir":
        vpr_ds = VPAir(largs.bd_args,ds_dir,ds_name,largs.data_split)
        # Not really useful, I guess :')
        vpr_distractor_ds = VPAir_Distractor(largs.bd_args,ds_dir,ds_name,largs.data_split)
    elif ds_name=="laurel_caverns":
        vpr_ds = Laurel(largs.bd_args,ds_dir,ds_name,largs.data_split)
    elif ds_name=="eiffel":
        vpr_ds = Eiffel(largs.bd_args,ds_dir,ds_name,largs.data_split)
    else:
        vpr_ds = BaseDataset(largs.bd_args, ds_dir, ds_name, 
                        largs.data_split)
    db_descs = torch.load(f"{pca_dir}/db-{ds_name}.pt")
    qu_descs = torch.load(f"{pca_dir}/qu-{ds_name}.pt")
    assert db_descs.shape == (vpr_ds.database_num, 
                                largs.pca_dim_reduce)
    assert qu_descs.shape == (vpr_ds.queries_num,
                                largs.pca_dim_reduce)
    print(f"Loaded: {db_descs.shape = }, {qu_descs.shape = }")
    
    print("----- Calculating recalls through top-k matching -----")
    dists, indices, recalls = get_top_k_recall(largs.top_k_vals, 
        db_descs, qu_descs, vpr_ds.soft_positives_per_query, 
        sub_sample_db=largs.sub_sample_db, 
        sub_sample_qu=largs.sub_sample_qu)
    print("------------ Recalls calculated ------------")
    
    print("--------------------- Results ---------------------")
    ts = time.strftime(f"%Y_%m_%d_%H_%M_%S")
    caching_directory = largs.prog.cache_dir
    results = {
        "Experiment-ID": str(largs.exp_id),
        "DB-Name": str(ds_name),
        "Num-DB": str(len(db_descs)),
        "Num-QU": str(len(qu_descs)),
        "Model-Type": str(largs.model_type),
        "Desc-Layer": str(largs.desc_layer),
        "Desc-Facet": str(largs.desc_facet),
        "Num-Clusters": str(largs.num_clusters),
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


# Entrypoint
if __name__ == "__main__" and "ipykernel" not in sys.argv[0]:
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
args = LocalArgs(ProgArgs(
        cache_dir="/scratch/avneesh.mishra/vl-vpr/cache",
        data_vg_dir="/home2/avneesh.mishra/Documents/vl-vpr/"\
                    "datasets_vg/datasets",
        vg_dataset_name="baidu_datasets"),
        pca_dim_reduce=512)

# %%
main(args)

# %%
