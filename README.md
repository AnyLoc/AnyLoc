# AnyLoc: Towards Universal Visual Place Recognition

[![License: BSD-3](https://img.shields.io/badge/License-BSD--3-yellow.svg?style=flat-square)](https://opensource.org/license/BSD-3-clause/)
[![stars](https://img.shields.io/github/stars/AnyLoc/AnyLoc?style=social)](https://github.com/AnyLoc/AnyLoc/stargazers)
[![arXiv](https://img.shields.io/badge/arXiv-2308.00688-b31b1b.svg)](https://arxiv.org/abs/2308.00688)
[![githubio](https://img.shields.io/badge/-anyloc.github.io-blue?logo=Github&color=grey)](https://anyloc.github.io/)
[![github](https://img.shields.io/badge/GitHub-Anyloc%2FAnyloc-blue?logo=Github)](https://github.com/AnyLoc/AnyLoc)
[![YouTube](https://img.shields.io/badge/YouTube-FF0000?style=flat&logo=youtube&logoColor=white)](https://youtu.be/ITo8rMInatk)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20HF%20Space-AnyLoc-blue)](https://huggingface.co/spaces/TheProjectsGuy/AnyLoc)
[![Open In Colab: Global Descriptors](https://img.shields.io/badge/IIITH--OneDrive-Global%20Descriptors-blue?logo=googlecolab&label=&labelColor=grey)](https://colab.research.google.com/github/AnyLoc/AnyLoc/blob/main/demo/anyloc_vlad_generate_colab.ipynb)
[![Open In Colab: Cluster visualizations](https://img.shields.io/badge/IIITH--OneDrive-Cluster%20Visualizations-blue?logo=googlecolab&label=&labelColor=grey)](https://colab.research.google.com/github/AnyLoc/AnyLoc/blob/main/demo/images_vlad_clusters.ipynb)
[![Public Release on IIITH-OneDrive](https://img.shields.io/badge/IIITH--OneDrive-Public%20Material-%23D83B01?logo=microsoftonedrive&logoColor=%230078D4&label=&labelColor=grey)][public-release-link]
[![Hugging Face Paper](https://img.shields.io/badge/%F0%9F%A4%97-HF--Paper-blue)](https://huggingface.co/papers/2308.00688)

[public-release-link]: https://iiitaphyd-my.sharepoint.com/:f:/g/personal/robotics_iiit_ac_in/EtpBLzBFfqdHljqQMnm6xdoBzW-4KFLXieXDVN4vPg84Lg?e=BP6ZW1

## Table of contents

- [AnyLoc: Towards Universal Visual Place Recognition](#anyloc-towards-universal-visual-place-recognition)
    - [Table of contents](#table-of-contents)
    - [Contents](#contents)
        - [Included Repositories](#included-repositories)
        - [Included Datasets](#included-datasets)
    - [PapersWithCode Badges](#paperswithcode-badges)
    - [Getting Started](#getting-started)
        - [Using the SOTA: AnyLoc-VLAD-DINOv2](#using-the-sota-anyloc-vlad-dinov2)
        - [Using the APIs](#using-the-apis)
            - [DINOv2](#dinov2)
            - [VLAD](#vlad)
            - [DINOv1](#dinov1)
    - [Validating the Results](#validating-the-results)
        - [NVIDIA NGC Singularity Container Setup](#nvidia-ngc-singularity-container-setup)
        - [Dataset Setup](#dataset-setup)
    - [References](#references)
        - [Cite Our Work](#cite-our-work)

## Contents

The contents of this repository are as follows

| S. No. | Item | Description |
| :---: | :--- | :----- |
| 1 | [demo](./demo/) | Contains standalone demo scripts (Quick start, Jupyter Notebook, and Gradio app) to run our `AnyLoc-VLAD-DINOv2` method. Also contains guides for APIs. This folder is self-contained (doesn't use anything outside it). |
| 2 | [scripts](./scripts/) | Contains all scripts for development. Use the `-h` option for argument information. |
| 3 | [configs.py](./configs.py) | Global configurations for the repository |
| 4 | [utilities](./utilities.py) | Utility Classes & Functions (includes DINOv2 hooks & VLAD) |
| 5 | [conda-environment.yml](./conda-environment.yml) | The conda environment (it could fail to install OpenAI CLIP as it includes a `git+` URL). We suggest you use the [setup_conda.sh](./setup_conda.sh) script. |
| 6 | [requirements.txt](./requirements.txt) | Requirements file for pip virtual environment. Probably out of date. |
| 7 | [custom_datasets](./custom_datasets/) | Custom datalaoder implementations for VPR. |
| 8 | [examples](./examples/) | Miscellaneous example scripts |
| 9 | [MixVPR](./MixVPR/) | Minimal MixVPR inference code |
| 10 | [clip_wrapper.py](./clip_wrapper.py) | A wrapper around two CLIP implementations (OpenAI and OpenCLIP). |
| 11 | [models_mae.py](./models_mae.py) | MAE implementation |
| 12 | [dino_extractor.py](./dino_extractor.py) | DINO (v1) feature extractor |
| 13 | [CONTRIBUTING.md](./CONTRIBUTING.md) | Note for contributors |
| 14 | [paper_utils](./paper_utils/) | Paper scripts (formatting for figures, etc.) |

### Included Repositories

Includes the following repositories (currently not submodules) as subfolders.

| Directory | Link | Cloned On | Description |
| :--- | :---- | :---- | :-----|
| [dvgl-benchmark](./dvgl-benchmark/) | [gmberton/deep-visual-geo-localization-benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark) | 2023-02-12 | For benchmarking |
| [datasets-vg](./datasets-vg/) | [gmberton/datasets_vg](https://github.com/gmberton/datasets_vg) | 2023-02-13 | For dataset download and formatting |
| [CosPlace](./CosPlace/) | [gmberton/CosPlace](https://github.com/gmberton/CosPlace) | 2023-03-20 | Baseline Comparisons |

### Included Datasets

We release all the benchmarking datasets in our [public release][public-release-link].

1. Download the `.tar.gz` files from [here][public-release-link] > `Datasets-All` (for the datasets you want to use)
2. Unzip them using `tar -xvzf ./NAME.tar.gz`. They should unzip into a directory with `NAME`.

    - If you're using our benchmarking [scripts](./scripts/), this directory where you're storing the datasets is the parameter `--prog.data-vg-dir` (in most scripts).
    - See [Dataset Setup](#dataset-setup) for detailed information (including how the data directory structure should look after unzipping)

We thank the following sources for the rich datasets

1. Baidu Autonomous Driving Business Unit for the Baidu Mall dataset present in `baidu_datasets.tar.gz`
2. Queensland University of Technology for the Gardens Point dataset present in `gardens.tar.gz`
3. York University for the 17 Places dataset present in `17places.tar.gz`
4. Tokyo Institute of Technology, INRIA, and CTU Prague for the Pitts-30k dataset present in `pitts30k.tar.gz`
5. Queensland University of Technology for the St. Lucia dataset present in `st_lucia.tar.gz`
6. University of Oxford for the Oxford dataset present in `Oxford_Robotcar.tar.gz`
7. AirLab at CMU for the Hawkins dataset present in `hawkins_long_corridor.tar.gz`, the Laurel Caverns dataset present in `laurel_caverns.tar.gz`, and the Nardo Air dataset present in `test_40_midref_rot0.tar.gz` (not rotated) and `test_40_midref_rot90.tar.gz` (rotated).
8. Fraunhofer FKIE and TU Munich for the VP-Air dataset present in `VPAir.tar.gz`
9. Ifremer and University of Toulon for the Mid-Atlantic Ridge dataset present in `eiffel.tar.gz`

Most of the contents of the zipped folders are from the original sources. We generate the ground truth for some of the datasets as `.npy` files; see [this issue](https://github.com/AnyLoc/AnyLoc/issues/8#issuecomment-1712450557) for more information.

The copyright of each dataset is held by the original sources.

## PapersWithCode Badges

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/anyloc-towards-universal-visual-place/visual-place-recognition-on-17-places)](https://paperswithcode.com/sota/visual-place-recognition-on-17-places?p=anyloc-towards-universal-visual-place)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/anyloc-towards-universal-visual-place/visual-place-recognition-on-baidu-mall)](https://paperswithcode.com/sota/visual-place-recognition-on-baidu-mall?p=anyloc-towards-universal-visual-place)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/anyloc-towards-universal-visual-place/visual-place-recognition-on-gardens-point)](https://paperswithcode.com/sota/visual-place-recognition-on-gardens-point?p=anyloc-towards-universal-visual-place)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/anyloc-towards-universal-visual-place/visual-place-recognition-on-hawkins)](https://paperswithcode.com/sota/visual-place-recognition-on-hawkins?p=anyloc-towards-universal-visual-place)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/anyloc-towards-universal-visual-place/visual-place-recognition-on-laurel-caverns)](https://paperswithcode.com/sota/visual-place-recognition-on-laurel-caverns?p=anyloc-towards-universal-visual-place)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/anyloc-towards-universal-visual-place/visual-place-recognition-on-mid-atlantic)](https://paperswithcode.com/sota/visual-place-recognition-on-mid-atlantic?p=anyloc-towards-universal-visual-place)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/anyloc-towards-universal-visual-place/visual-place-recognition-on-nardo-air)](https://paperswithcode.com/sota/visual-place-recognition-on-nardo-air?p=anyloc-towards-universal-visual-place)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/anyloc-towards-universal-visual-place/visual-place-recognition-on-nardo-air-r)](https://paperswithcode.com/sota/visual-place-recognition-on-nardo-air-r?p=anyloc-towards-universal-visual-place)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/anyloc-towards-universal-visual-place/visual-place-recognition-on-oxford-robotcar-4)](https://paperswithcode.com/sota/visual-place-recognition-on-oxford-robotcar-4?p=anyloc-towards-universal-visual-place)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/anyloc-towards-universal-visual-place/visual-place-recognition-on-vp-air)](https://paperswithcode.com/sota/visual-place-recognition-on-vp-air?p=anyloc-towards-universal-visual-place)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/anyloc-towards-universal-visual-place/visual-place-recognition-on-st-lucia)](https://paperswithcode.com/sota/visual-place-recognition-on-st-lucia?p=anyloc-towards-universal-visual-place)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/anyloc-towards-universal-visual-place/visual-place-recognition-on-pittsburgh-30k)](https://paperswithcode.com/sota/visual-place-recognition-on-pittsburgh-30k?p=anyloc-towards-universal-visual-place)

## Getting Started

> **Tip**: You can explore the [HuggingFace Space](https://huggingface.co/spaces/TheProjectsGuy/AnyLoc) and the Colab notebooks (no GPU needed).

Clone this repository

```bash
git clone https://github.com/AnyLoc/AnyLoc.git
cd AnyLoc
```

Set up the conda environment (you can also use `mamba` instead of `conda`; the script will automatically detect it)

```bash
conda create -n anyloc python=3.9
conda activate anyloc
bash ./setup_conda.sh
# If you want to install the developer tools as well
bash ./setup_conda.sh --dev
```

The setup takes about 11 GB of disk space.
You can also use an existing conda environment, say `vl-vpr`, by doing

```bash
bash ./setup_conda.sh vl-vpr
```

Note the following:

- All our public release files can be found in our [public release][public-release-link].
    - If the conda environment setup is taking time, you could just unzip `conda-env.tar.gz` (GB) in your `~/anaconda3/envs` folder (but compatibility is not guaranteed).
- The `./scripts` folder is for validating our results and seeing the main scripts. Most applications are in the `./demo` folder. See the list of [demos](./demo/) before running anything.
- If you're running something in the `./scripts` folder, run it with `pwd` in this (repository) folder. For example, python scripts are run as `python ./scripts/<script>.py` and bash scripts are run as `bash ./scripts/<script>.sh`. For the demos and other baselines, you should `cd` into respective folders.
- The [utilities.py](./utilities.py) file is mainly for `./scripts` files. All demos actually use the [demo/utilities.py](./demo/utilities.py) file (which is distilled and minimal). Using the latter should be enough to implement our SOTA method.

### Using the SOTA: AnyLoc-VLAD-DINOv2

[![Open In Colab](https://img.shields.io/badge/IIITH--OneDrive-Global%20Descriptors-blue?logo=googlecolab&label=&labelColor=grey)](https://colab.research.google.com/github/AnyLoc/AnyLoc/blob/main/demo/anyloc_vlad_generate_colab.ipynb)
[![Local Python script](https://img.shields.io/badge/Python-.%2Fdemo%2Fanyloc__vlad__generate.py-blue?logo=python&labelColor=white&label=&color=gray)](./demo/anyloc_vlad_generate.py)

### Using the APIs

Import the utilities

```py
from utilities import DinoV2ExtractFeatures
from utilities import VLAD
```

#### DINOv2

DINOv2 feature extractor can be used as follows

```py
extractor = DinoV2ExtractFeatures("dinov2_vitg14", desc_layer,
        desc_facet, device=device)
```

Get the descriptors using

```py
# Make image patchable (14, 14 patches)
c, h, w = img_pt.shape
h_new, w_new = (h // 14) * 14, (w // 14) * 14
img_pt = tvf.CenterCrop((h_new, w_new))(img_pt)[None, ...]
# Main extraction
ret = extractor(img_pt) # [1, num_patches, desc_dim]
```

#### VLAD

The VLAD aggregator can be loaded with vocabulary (cluster centers) from a `c_centers.pt` file.

```py
# Main VLAD object
vlad = VLAD(num_c, desc_dim=None, cache_dir=os.path.dirname(c_centers_file))
vlad.fit(None)  # Load the vocabulary (and auto-detect `desc_dim`)
# Cluster centers have shape: [num_c, desc_dim]
#   - num_c: number of clusters
#   - desc_dim: descriptor dimension
```

If you have a database of descriptors you want to fit, use

```py
vlad.fit(ein.rearrange(full_db_vlad, "n k d -> (n k) d"))
# n: number of images
# k: number of patches/descriptors per image
# d: descriptor dimension
```

To get the VLAD representations of multiple images, use

```py
db_vlads: torch.Tensor = vlad.generate_multi(full_db)
# Shape of full_db: [n_db, n_d, d_dim]
#   - n_db: number of images in the database
#   - n_d: number of descriptors per image
#   - d_dim: descriptor dimension
# Shape of db_vlads: [n_db, num_c * d_dim]
#   - num_c: number of clusters (centers)
```

#### DINOv1

This is present in [dino_extractor.py](./dino_extractor.py) (not a part of [demo/utilities.py](./demo/utilities.py)).

Initialize and use it as follows the extractor

```py
# Import it
from dino_extractor import ViTExtractor
...

# Initialize it (layer and key are when extracting descriptors)
extractor = ViTExtractor("dino_vits8", stride=4, 
        device=device)
...

# Use it to extract patch descriptors
img = ein.rearrange(img, "c h w -> 1 c h w").to(device)
img = F.interpolate(img, (224, 298))    # For 4:3 images
desc = extractor.extract_descriptors(img,
        layer=11, facet="key") # [1, 1, num_descs, d_dim]
...
```

## Validating the Results

You don't need to read further if you're not experimentally validating the entire results (enjoy the [demos](./demo/) instead) or building on this repository from source.

The following sections are for the curious minds who want to reproduce the results.

> Note to/for contributors: Please follow [contributing guidelines](./CONTRIBUTING.md). This is mainly for developers/authors who'll be pushing to this repository.

All the runs were done on a machine with the following specifications:

- CPU: Two Intel Xeon Gold 5317 CPUs (12C24T each)
- CPU RAM: 256 GB
- GPUs: Four NVIDIA RTX 3090 GPUs (24 GB, 10496 CUDA cores each)
- Storage: 3.5 TB HDD on `/scratch`. However, all datasets will take 32+ GB, have more for other requirements (for VLAD cluster centers, caching, models, etc.). We noticed that singularity (with SIF, cache, and tmp) used 90+ GB.
    - Driver Version (NVIDIA-SMI): 570.47.03
    - CUDA (SMI): 11.6

We use only one GPU; however, some experiments (with large datasets) might need all of the CPU RAM (for efficient/fast nearest neighbor search). Ideally, a 16 GB GPU should also work.

Do the following

1. Clone the repository and setup the NVIDIA NGC container (run everything inside it)
2. Setup the datasets (download, format, and extract them)
3. Run the script you want to test from [scripts](./scripts/) folder

Start by cloning/setting up the repository

```bash
cd ~/Documents
git clone https://github.com/AnyLoc/AnyLoc.git vl-vpr
```

### NVIDIA NGC Singularity Container Setup

Despite using [recommended practices of reproducibility](https://pytorch.org/docs/stable/notes/randomness.html) (see function `seed_everything` in [utilities.py](./utilities.py)) in PyTorch, we noticed minor changes across GPU types and CUDA versions. To mitigate this, we recommend using a singularity container.

Setting up the environment in a singularity container (in a SLURM environment)

> **TL;DR**: Run the following (this system is a different one). This was tested on [CMU's Bridges-2 partition of PSC HPC](https://www.psc.edu/resources/bridges-2/). Don't use this if you want to replicate the tables in the paper (but the numbers come close).

```bash
salloc -p GPU-small -t 01:00:00 --ntasks-per-node=5 --gres=gpu:v100-32:1
cd /ocean/containers/ngc/pytorch/
singularity instance start --nv pytorch_22.12-py3.sif vlvpr
singularity run --nv instance://vlvpr
cd /ocean/projects/cis220039p/nkeetha/data/singularity/venv
source vlvpr/bin/activate
cd /ocean/projects/cis220039p/<path to vl-vpr scripts folder>
```

> **Main setup**: For Singularity on [IIITH's Ada HPC](https://hpc.iiit.ac.in/wiki/index.php/Ada_User_Guide) (Ubuntu 18.04) - our main setup for validation. Use this if you want to replicate the tables in the paper (hardware should be same as listed before).

The script below assumes that this repository is cloned in `~/Documents/vl-vpr`. That is, this README is at `~/Documents/vl-vpr/README.md`.

```bash
# Load the module and configurations
module load u18/singularity-ce/3.9.6
mkdir -p /scratch/$USER/singularity && cd $_ && mkdir .cache .tmp venvs
export SINGULARITY_CACHEDIR=/scratch/$USER/singularity/.cache
export SINGULARITY_TMPDIR=/scratch/$USER/singularity/.tmp
# Ensure that the next command gives output "1" (or anything other than "0")
cat /proc/sys/kernel/unprivileged_userns_clone
# Setup the container (download the image if not there already) - (15 GB cache + 7.5 GB file)
singularity pull ngc_pytorch_22.12-py3 docker://nvcr.io/nvidia/pytorch:22.12-py3
# Test container through shell
singularity shell --nv ngc_pytorch_22.12-py3
# Start and run the container (mount the symlinked and scratch folders)
singularity instance start --mount "type=bind,source=/scratch/$USER,destination=/scratch/$USER" \
    --nv ngc_pytorch_22.12-py3 vl-vpr
singularity run --nv instance://vl-vpr
# Create virtual environment
cd ~/Documents/vl-vpr/
pip install virtualenv
cd venvs
virtualenv --system-site-packages vl-vpr
# Activate virtualenv and install all packages
cd ~/Documents/vl-vpr/
source ./venvs/vl-vpr/bin/activate
bash ./setup_virtualenv_ngc.sh
# Run anything you want (from here, but find the file in scripts)
cd ~/Documents/vl-vpr/
python ./scripts/<task name>.py <args>
# The baseline scripts should be run in their own folders. For example, to run CosPlace, do
cd ~/Documents/vl-vpr/
cd CosPlace
python ./<script>.py
```

### Dataset Setup

> **Datasets Note**: See the `Datasets-All` folder in our [public material][public-release-link] (for `.tar.gz` files). Also see [included datasets](#included-datasets).

Set them up in a folder with sufficient space

```bash
mkdir -p /scratch/$USER/vl-vpr/datasets && cd $_
```

Download (and unzip) the datasets from [here][public-release-link] (`Datasets-All` folder) into this folder. Link this folder (for easy access form this repository)

```bash
cd ~/Documents/vl-vpr/
cd ./datasets-vg
ln -s /scratch/$USER/vl-vpr/datasets datasets
```

After setting up all datasets, the folders should look like this (in the dataset folder). Run the following command to get the tree structure.

```bash
tree ./eiffel ./hawkins*/ ./laurel_caverns ./VPAir ./test_40_midref_rot*/ ./Oxford_Robotcar ./gardens ./17places ./baidu_datasets ./st_lucia ./pitts30k --filelimit=20 -h
```

- The `test_40_midref_rot0` is `Nardo Air`. This is also referred as `Tartan_GNSS_notrotated` in our scripts.
- The `test_40_midref_rot90` is `Nardo Air-R` (rotated). This is also referred as `Tartan_GNSS_rotated` in out scripts.
- The `hawkins_long_corridor` is the Hawkins dataset (degraded environment).
- The `eiffel` dataset is `Mid-Atlantic Ridge` (underwater dataset).

Output will be something like

```txt
./eiffel
├── [4.0K]  db_images [65 entries exceeds filelimit, not opening dir]
├── [2.2K]  eiffel_gt.npy
└── [4.0K]  q_images [101 entries exceeds filelimit, not opening dir]
./hawkins_long_corridor/
├── [4.0K]  db_images [127 entries exceeds filelimit, not opening dir]
├── [ 12K]  images [314 entries exceeds filelimit, not opening dir]
├── [ 17K]  pose_topic_list.npy
└── [4.0K]  q_images [118 entries exceeds filelimit, not opening dir]
./laurel_caverns
├── [4.0K]  db_images [141 entries exceeds filelimit, not opening dir]
├── [ 20K]  images [744 entries exceeds filelimit, not opening dir]
├── [ 41K]  pose_topic_list.npy
└── [4.0K]  q_images [112 entries exceeds filelimit, not opening dir]
./VPAir
├── [ 677]  camera_calibration.yaml
├── [420K]  distractors [10000 entries exceeds filelimit, not opening dir]
├── [4.0K]  distractors_temp
├── [ 321]  License.txt
├── [177K]  poses.csv
├── [ 72K]  queries [2706 entries exceeds filelimit, not opening dir]
├── [160K]  reference_views [2706 entries exceeds filelimit, not opening dir]
├── [ 96K]  reference_views_npy [2706 entries exceeds filelimit, not opening dir]
└── [ 82K]  vpair_gt.npy
./test_40_midref_rot0/
├── [ 46K]  gt_matches.csv
├── [2.8K]  network_config_dump.yaml
├── [5.3K]  query.csv
├── [4.0K]  query_images [71 entries exceeds filelimit, not opening dir]
├── [2.9K]  reference.csv
└── [4.0K]  reference_images [102 entries exceeds filelimit, not opening dir]
./test_40_midref_rot90/
├── [ 46K]  gt_matches.csv
├── [2.8K]  network_config_dump.yaml
├── [5.3K]  query.csv
├── [4.0K]  query_images [71 entries exceeds filelimit, not opening dir]
├── [2.9K]  reference.csv
└── [4.0K]  reference_images [102 entries exceeds filelimit, not opening dir]
./Oxford_Robotcar
├── [4.0K]  __MACOSX
│   └── [4.0K]  oxDataPart
├── [4.0K]  oxDataPart
│   ├── [4.0K]  1-m [191 entries exceeds filelimit, not opening dir]
│   ├── [ 24K]  1-m.npz
│   ├── [ 13K]  1-m.txt
│   ├── [4.0K]  1-s [191 entries exceeds filelimit, not opening dir]
│   ├── [ 24K]  1-s.npz
│   ├── [4.0K]  1-s-resized [191 entries exceeds filelimit, not opening dir]
│   ├── [ 13K]  1-s.txt
│   ├── [4.0K]  2-s [191 entries exceeds filelimit, not opening dir]
│   ├── [ 24K]  2-s.npz
│   ├── [4.0K]  2-s-resized [191 entries exceeds filelimit, not opening dir]
│   └── [ 13K]  2-s.txt
├── [ 15K]  oxdatapart.mat
└── [ 66M]  oxdatapart_seg.npz
./gardens
├── [4.0K]  day_left [200 entries exceeds filelimit, not opening dir]
├── [4.0K]  day_right [200 entries exceeds filelimit, not opening dir]
├── [3.6K]  gardens_gt.npy
└── [4.0K]  night_right [200 entries exceeds filelimit, not opening dir]
./17places
├── [ 14K]  ground_truth_new.npy
├── [ 13K]  my_ground_truth_new.npy
├── [ 12K]  query [406 entries exceeds filelimit, not opening dir]
├── [ 514]  ReadMe.txt
└── [ 12K]  ref [406 entries exceeds filelimit, not opening dir]
./baidu_datasets
├── [4.0G]  IDL_dataset_cvpr17_3852.zip
├── [387M]  mall.pcd
├── [108K]  query_gt [2292 entries exceeds filelimit, not opening dir]
├── [ 96K]  query_images_undistort [2292 entries exceeds filelimit, not opening dir]
├── [2.7K]  readme.txt
├── [ 44K]  training_gt [689 entries exceeds filelimit, not opening dir]
└── [ 36K]  training_images_undistort [689 entries exceeds filelimit, not opening dir]
./st_lucia
├── [4.0K]  images
│   └── [4.0K]  test
│       ├── [180K]  database [1549 entries exceeds filelimit, not opening dir]
│       └── [184K]  queries [1464 entries exceeds filelimit, not opening dir]
└── [695K]  map_st_lucia.png
./pitts30k
└── [4.0K]  images
    ├── [4.0K]  test
    │   ├── [1.2M]  database [10000 entries exceeds filelimit, not opening dir]
    │   ├── [5.9M]  database.npy
    │   ├── [864K]  queries [6816 entries exceeds filelimit, not opening dir]
    │   └── [4.0M]  queries.npy
    ├── [4.0K]  train
    │   ├── [1.3M]  database [10000 entries exceeds filelimit, not opening dir]
    │   ├── [5.9M]  database.npy
    │   ├── [948K]  queries [7416 entries exceeds filelimit, not opening dir]
    │   └── [4.4M]  queries.npy
    └── [4.0K]  val
        ├── [1.3M]  database [10000 entries exceeds filelimit, not opening dir]
        ├── [5.8M]  database.npy
        ├── [980K]  queries [7608 entries exceeds filelimit, not opening dir]
        └── [4.4M]  queries.npy
```

These directories are put under `./datasets_vg/datasets` folder (can store them in scratch and symlink it there). For example, the 17places dataset can be found under `./datasets_vg/datasets/17places` folder.

Original dataset webpages:

- [Oxford RobotCar](https://robotcar-dataset.robots.ox.ac.uk/)
- [St. Lucia](https://open.qcr.ai/dataset/st-lucia-multiple/) (also see other datasets in [VPR-Bench](https://open.qcr.ai/dataset/vprbench/))
- [Pitts 30k](http://www.ok.ctrl.titech.ac.jp/~torii/project/repttile/) (could also find it [here](https://github.com/Relja/netvlad/issues/37))
- [Gardens Point](https://zenodo.org/record/4590133)
- [17places](https://www.raghavendersahdev.com/place-recognition.html) (zipped download [link](https://cloudstor.aarnet.edu.au/plus/s/6sMAG8djfQvWuIx))
- [Baidu Mall](https://openaccess.thecvf.com/content_cvpr_2017/html/Sun_A_Dataset_for_CVPR_2017_paper.html) (zipped on [dropbox from authors](https://www.dropbox.com/s/4mksiwkxb7t4a8a/IDL_dataset_cvpr17_3852.zip?dl=0))
- [VPAir](https://github.com/AerVisLoc/vpair)
- [Mid-Atlantic Ridge](https://www.seanoe.org/data/00680/79218/)
- [Hawkins, Laurel Caverns, and Nardo Air](https://drive.google.com/drive/u/1/folders/1CweSoePAxo7znoHMJmy5Ntn3CJqQrZ_u)

Some datasets can be found at other places

- [SuperOdometry - ICCV 2023 SLAM Challenge](https://superodometry.com/datasets)
- [ZhangXiwuu/Awesome_visual_place_recognition_datasets](https://github.com/ZhangXiwuu/Awesome_visual_place_recognition_datasets)

## References

We thank the authors of the following repositories for their open source code and data:

- VPR Datasets
    - [gmberton/datasets_vg](https://github.com/gmberton/datasets_vg): Downloading and formatting
- Baselines
    - [gmberton/deep-visual-geo-localization-benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark)
    - [gmberton/CosPlace](https://github.com/gmberton/CosPlace)
    - [amaralibey/MixVPR](https://github.com/amaralibey/MixVPR)
- CLIP
    - [openai/CLIP](https://github.com/openai/CLIP): Official CLIP implementation
    - [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip): Open source implementation of CLIP with more checkpoints
- DINO
    - [facebookresearch/dino](https://github.com/facebookresearch/dino)
    - [ShirAmir/dino-vit-features](https://github.com/ShirAmir/dino-vit-features)
- [SAM](https://github.com/facebookresearch/segment-anything)
- [MAE](https://github.com/facebookresearch/mae)
- [DINOv2](https://github.com/facebookresearch/dinov2)

### Cite Our Work

Thanks for using our work. You can cite it as:

```bib
@article{AnyLoc,
    author    = {Nikhil Keetha and Avneesh Mishra and Jay Karhade and Krishna Murthy Jatavallabhula and Sebastian Scherer and Madhava Krishna and Sourav Garg}
    title     = {AnyLoc: Towards Universal Visual Place Recognition},
    url       = {https://arxiv.org/abs/2308.00688}
    journal   = {arXiv},
    year      = {2023},
}
```

Developers:

- [TheProjectsGuy](https://github.com/TheProjectsGuy) ([Avneesh Mishra](https://theprojectsguy.github.io/))
- [JayKarhade](https://github.com/JayKarhade) ([Jay Karhade](https://jaykarhade.netlify.app/))
- [Nik-V9](https://github.com/Nik-V9) ([Nikhil Keetha](https://nik-v9.github.io/))
- [krrish94](https://github.com/krrish94) ([Krishna Murthy Jatavallabhula](https://krrish94.github.io/))
- [oravus](https://github.com/oravus) ([Sourav Garg](https://scholar.google.co.in/citations?user=oVS3HHIAAAAJ&hl=en))
