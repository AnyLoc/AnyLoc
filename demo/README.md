# Demo and Quick Start for AnyLoc

## Table of Contents

- [Demo and Quick Start for AnyLoc](#demo-and-quick-start-for-anyloc)
    - [Table of Contents](#table-of-contents)
    - [Contents](#contents)
    - [Before you start](#before-you-start)
    - [Included Demos](#included-demos)
        - [Demo 1: Global Descriptors](#demo-1-global-descriptors)
        - [Demo 2: VLAD Cluster assignment visualization](#demo-2-vlad-cluster-assignment-visualization)

## Contents

| S. No | File Name | Description |
| :--- | :--- | :--- |
| 1 | [anyloc_vlad_generate_colab.ipynb](./anyloc_vlad_generate_colab.ipynb) | Jupyter notebook (for Google Colab) to generate global descriptors using AnyLoc-VALD-DINOv2. |
| 2 | [anyloc_vlad_generate.py](./anyloc_vlad_generate.py) | Python script to generate global descriptors using AnyLoc-VALD-DINOv2. |
| 3 | [images_vlad_colab.ipynb](./images_vlad_colab.ipynb) | WIP: Jupyter notebook (for Google Colab). Visualize cluster assignments. |
| 4 | [hf_imgs_vlad_clusters.py](./hf_imgs_vlad_clusters.py) | WIP: HuggingFace app for visualizing the cluster assignments. |

## Before you start

You'll need the following before getting started. The Colab notebooks actually setup all this (as a part of the notebook), they're fully **standalone**.

1. **Cluster Centers**: Download the cluster centers from the [public data](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/avneesh_mishra_research_iiit_ac_in/Ek6y97czRqRIgIrd4Yj2_aYBz02Nkvmdbh_9Ec_-HgMSHw): `Colab1 > cache.zip`. Unzip the file in `pwd` with `unzip cache.zip` so that `./cache` is created here. Vocabulary is stored in `c_center.pt` files in folders like `./cache/vocabulary/dinov2_vitg14/l31_value_c32/urban`.

## Included Demos

### Demo 1: Global Descriptors

[![Open In Colab](https://img.shields.io/badge/IIITH--OneDrive-Global%20Descriptors-blue?logo=googlecolab&label=&labelColor=grey)](https://colab.research.google.com/github/AnyLoc/AnyLoc/blob/main/demo/anyloc_vlad_generate_colab.ipynb)
[![Local Python script](https://img.shields.io/badge/Python-anyloc__vlad__generate.py-blue?logo=python&labelColor=white&label=&color=gray)](./anyloc_vlad_generate.py)

If you want to use the python script (after cloning the repo and setting up utilities).

```bash
python ./anyloc_vlad_generate.py
# Or if you have your own *.png images
python ./anyloc_vlad_generate.py --in-dir ./data/images --imgs-ext png --out-dir ./data/descriptors
```

Use `--domain` to specify the domain of the images (default is `urban`). Use `--help` to see all the options.

> **Tip**: You can get an idea of which domain to use by using our [HuggingFace Space](https://huggingface.co/spaces/TheProjectsGuy/AnyLoc) app. Upload a representative sample of your images under `GeM t-SNE Projection` and see which group of images do your uploaded images come close to.

### Demo 2: VLAD Cluster assignment visualization

[![Open In Colab: Cluster visualizations](https://img.shields.io/badge/IIITH--OneDrive-Cluster%20Visualizations-blue?logo=googlecolab&label=&labelColor=grey)](https://colab.research.google.com/github/AnyLoc/AnyLoc/blob/main/demo/images_vlad_clusters.ipynb)
