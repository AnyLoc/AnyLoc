# AnyLoc: Towards Universal Visual Place Recognition

[![License: BSD-3](https://img.shields.io/badge/License-BSD--3-yellow.svg?style=flat-square)](https://opensource.org/license/BSD-3-clause/)
[![stars](https://img.shields.io/github/stars/AnyLoc/AnyLoc?style=social)](https://github.com/AnyLoc/AnyLoc/stargazers)
[![arXiv](https://img.shields.io/badge/arXiv-2308.00688-b31b1b.svg)](https://arxiv.org/abs/2308.00688)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AnyLoc/AnyLoc/blob/main/demo/images_vlad_clusters.ipynb)
[![YouTube](https://img.shields.io/badge/YouTube-FF0000?style=flat&logo=youtube&logoColor=white)](https://youtu.be/ITo8rMInatk)
[![Website](https://img.shields.io/badge/Website-black)](https://anyloc.github.io/)

> **Note**: Work in progress. Major changes expected.

## Table of contents

- [AnyLoc: Towards Universal Visual Place Recognition](#anyloc-towards-universal-visual-place-recognition)
    - [Table of contents](#table-of-contents)
    - [Contents](#contents)
        - [Included Repositories](#included-repositories)
    - [PapersWithCode Badges](#paperswithcode-badges)
    - [Developer Setup](#developer-setup)
        - [HPC Setup](#hpc-setup)
            - [PSC Setup](#psc-setup)
            - [Ada Setup](#ada-setup)
        - [Dataset Setup](#dataset-setup)
    - [References](#references)

## Contents

The contents of this repository as as follows

| S. No. | Item | Description |
| :---: | :--- | :----- |
| 1 | [scripts](./scripts/) | Contains all scripts for development. Use the `-h` option for argument information. |
| 2 | [demo](./demo/) | Contains standalone demo scripts (Jupyter Notebook and Gradio app) to run our `AnyLoc-VLAD-DINOv2` method. |
| 3 | [configs.py](./configs.py) | Global configurations for the repository |
| 4 | [utilities](./utilities.py) | Utility Classes & Functions (includes DINOv2 hooks & VLAD) |
| 5 | [conda-environment.yml](./conda-environment.yml) | The conda environment (it could fail to install OpenAI CLIP as it includes a `git+` URL). We suggest you use the [setup_conda.sh](./setup_conda.sh) script. |
| 6 | [requirements.txt](./requirements.txt) | Requirements file for pip virtual environment. Potentially out of date. |
| 7 | [custom_datasets](./custom_datasets/) | Custom datalaoder implementations for VPR |
| 8 | [examples](./examples/) | Miscellaneous example scripts |
| 9 | [MixVPR](./MixVPR/) | Minimal MixVPR inference code |
| 10 | [clip_wrapper.py](./clip_wrapper.py) | A wrapper around two CLIP implementations |
| 11 | [models_mae.py](./models_mae.py) | MAE implementation |
| 12 | [dino_extractor.py](./dino_extractor.py) | DINO feature extractor |

### Included Repositories

Includes the following repositories (currently not submodules) as subfolders.

| Directory | Link | Cloned On | Description |
| :--- | :---- | :---- | :-----|
| [dvgl-benchmark](./dvgl-benchmark/) | [gmberton/deep-visual-geo-localization-benchmark](https://github.com/gmberton/deep-visual-geo-localization-benchmark) | 2023-02-12 | For benchmarking |
| [datasets-vg](./datasets-vg/) | [gmberton/datasets_vg](https://github.com/gmberton/datasets_vg) | 2023-02-13 | For dataset download and formatting |
| [CosPlace](./CosPlace/) | [gmberton/CosPlace](https://github.com/gmberton/CosPlace) | 2023-03-20 | Baseline Comparisons |

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

## Developer Setup

> [Note to contributors](CONTRIBUTING.md)

### HPC Setup

Setting up the environment in a singularity container (in a SLURM environment)

#### PSC Setup

For CMU

```bash
salloc -p GPU-small -t 01:00:00 --ntasks-per-node=5 --gres=gpu:v100-32:1
cd /ocean/containers/ngc/pytorch/
singularity instance start --nv pytorch_22.12-py3.sif vlvpr
singularity run --nv instance://vlvpr
cd /ocean/projects/cis220039p/nkeetha/data/singularity/venv
source vlvpr/bin/activate
cd /ocean/projects/cis220039p/<path to vl-vpr scripts folder>
```

#### Ada Setup

For Singularity on IIITH's Ada HPC

```bash
# Load the module and configurations
module load u18/singularity-ce/3.9.6
mkdir /scratch/$USER/singularity && cd $_ && mkdir .cache .tmp venvs
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
cd ~/Documents/vl-vpr/
# Create virtual environment
pip install virtualenv
cd venvs
virtualenv --system-site-packages vl-vpr
# Activate virtualenv and install all packages
cd ~/Documents/vl-vpr/
source ./venvs/vl-vpr/bin/activate
bash ./setup_virtualenv_ngc.sh
# Run anything you want (from here, but in scripts)
cd ~/Documents/vl-vpr/
python ./scripts/<task name>.py <args>
```

### Dataset Setup

After setting up the unstructured datasets, the folders should look like this (in the dataset folder)

```bash
tree ./eiffel ./hawkins*/ ./laurel_caverns ./VPAir ./test_40_midref_rot*/ ./Oxford_Robotcar ./gardens ./17places ./baidu_datasets ./st_lucia ./pitts30k --filelimit=20 -h
```

- The `test_40_midref_rot0` is `Nardo Air`. This is also referred as `Tartan_GNSS_notrotated` in our scripts.
- The `test_40_midref_rot90` is `Nardo Air-R` (rotated). This is also referred as `Tartan_GNSS_rotated` in out scripts.
- The `hawkins_long_corridor` is the Hawkins dataset.

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

Developers:

- [TheProjectsGuy](https://github.com/TheProjectsGuy) ([Avneesh Mishra](https://theprojectsguy.github.io/))
- [JayKarhade](https://github.com/JayKarhade) ([Jay Karhade](https://jaykarhade.netlify.app/))
- [Nik-V9](https://github.com/Nik-V9) ([Nikhil Keetha](https://nik-v9.github.io/))
- [krrish94](https://github.com/krrish94) ([Krishna Murthy Jatavallabhula](https://krrish94.github.io/))
- [oravus](https://github.com/oravus) ([Sourav Garg](https://scholar.google.co.in/citations?user=oVS3HHIAAAAJ&hl=en))
