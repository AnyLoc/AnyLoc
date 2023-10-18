# Global configurations
"""
"""

# %%
# Get the repository path as 'lib_path'
import os
import sys
from pathlib import Path
dir_name = None
try:
    dir_name = os.path.dirname(os.path.realpath(__file__))
except NameError:
    dir_name = os.path.abspath('')
lib_path = os.path.realpath(f"{Path(dir_name)}")
# Add to path
print("[INFO]: Configs is modifying path")
if lib_path not in sys.path:
    print(f"Adding library path: {lib_path} to PYTHONPATH")
    sys.path.append(lib_path)
else:
    print(f"Library path {lib_path} already in PYTHONPATH")
# Username on system
user_name = os.environ.get("USER", "avneesh.mishra")

# %%
# Import everything
import tyro
from dataclasses import dataclass, field
from typing import Literal, List, Union
import torch

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

# %%
def try_tyro(x, allow_safe_quit=False):
    """
        Wrap 'x' dataclass around tyro.cli (if it works)
        
        Parameters:
        - x:    A class wrapped in `dataclasses.dataclass` for tyro
        - allow_safe_quit:  If exit code is 0 (maybe `-h`), then exit
    """
    try:
        return tyro.cli(x)
    except (SystemExit, Exception) as exc:
        print("[WARN]: Tyro might not have parsed all arguments! "\
            f"Exception: {exc}")
        print("[INFO]: Ignore above warning if multiple tyro used")
        if str(exc) == "0":
            print("[DEBUG]: Exit code 0 received")
            if allow_safe_quit:
                print("[INFO]: Safe exit is enabled, exiting...")
                quit(0)
        return x()  # Passthrough


# %% --------------------- Simple variables ---------------------
@dataclass(frozen=True)
class ProgArgs:
    """
        Core program arguments
    """
    # A directory for storing cache
    cache_dir: Path = "/ocean/projects/cis220039p/jkarhade/data/vlvpr_cache"#f"/scratch/{user_name}/vl-vpr/cache"
    """
        Directory for cache storage. Values:
        - Avneesh: f"/scratch/{user_name}/vl-vpr/cache"
        - Jay: '/home/jay/Documents/vl-vpr'
    """
    # The directory where 'datasets-vg' are downloaded
    data_vg_dir: Path = "/ocean/projects/cis220039p/shared/datasets/vpr/datasets_vg"#f"{lib_path}/datasets_vg/datasets"
    """
        Directory where the datasets are stored. Values:
        - Avneesh: f"{lib_path}/datasets_vg/datasets"
        - Jay: '/home/jay/Downloads/vl_vpr_datasets'
    """
    # Default dataset name for VPR
    vg_dataset_name: Literal["st_lucia", "pitts30k", "17places", "nordland", "tokyo247", "baidu_datasets", "Oxford", "Oxford_25m", "gardens", "hawkins","hawkins_long_corridor", "global", "VPAir", "Tartan_GNSS_rotated", "Tartan_GNSS_notrotated", "Tartan_GNSS_test_notrotated", "Tartan_GNSS_test_rotated",  "laurel_caverns","eiffel"] = "hawkins_long_corridor"
    # Use wandb (False = No WandB)
    use_wandb: bool = False
    # WandB project name
    wandb_proj: str = "Baselines"
    # WandB entity (should be 'vpr-vl')
    wandb_entity: str = "vpr-vl"
    # WandB result group (within the project)
    wandb_group: str = "Oxford"
    # WandB run name (within the group)
    wandb_run_name: str = "Oxford/CLIP_TopK"
    # Save qualitative results for WandB
    wandb_save_qual: bool = False

prog_args = ProgArgs()
"""
    > Note: Default placeholder, not effected by tyro.
"""
_real_path = lambda x: os.path.realpath(os.path.expanduser(x))

# Cache folder for results
caching_directory = _real_path(prog_args.cache_dir)
"""
    A folder that has a lot of space (to store cache). The folder 
    structure is created by the program that uses this variable.
    
    > Note: Default placeholder, not effected by tyro.
"""

# Datasets directory
datasets_dir = _real_path(prog_args.data_vg_dir)
"""
    The folder where all the VPR datasets are stored in the format of 
    the `datasets-vg` repository. The format is usually of the form:\n
    datasets\n
    └── st_lucia\n
        ├── images\n
        │   └── test\n
        │       ├── database [1549 entries ...]\n
        │       └── queries [1464 entries ...]\n
        └── map_st_lucia.png\n
    Name `st_lucia` is just an example.
    
    > Note: Default placeholder, not effected by tyro.
"""

# Dataset name
dataset_name = prog_args.vg_dataset_name
"""
    Dataset name for VPR
    
    > Note: Default placeholder, not effected by tyro.
"""

# %% --------------------- Argument classes ---------------------
# For parsing datasets
@dataclass(frozen=True)
class BaseDatasetArgs:
    """
        Dataset arguments for BaseDataset in `datasets_ws.py`
    """
    # Resize shape: [H, W]
    resize: List[int] = field(default_factory=lambda:[480, 640])
    # Pre/post-processing methods and prediction refinement
    test_method: Literal["hard_resize", "single_query", \
            "central_crop", "five_crops", "nearest_crop", \
            "maj_voting"] = "hard_resize"
    """
        Pre/post processing method must be one of the following (str)
        - hard_resize:  Apply straightforward resize to above shape.
        - single_query: Resize to `min(resize)`. Used when queries 
                        have varying sizes and can't be batched.
        - central_crop: Take the biggest central crop of size resize. 
                        Preserves ratio.
        - five_crops:   See [1]
        - nearest_crop: See [1]
        - maj_voting:   See [1]
        
        [1]:            Get 5 square crops with size `min(resize)` and
                        batch them.
    """
    # Threshold value for positive distance (classification)
    val_positive_dist_threshold: int = 25

base_dataset_args = BaseDatasetArgs()
"""
    > Note: Default placeholder, not effected by tyro.
"""


# %%
# Experimental section

# %%


# %%
# Experimental section

# %%
