# Scripts for exploring foundation models based VPR

> **Note** to end users: Scripts in this folder are for development purposes only. These aren't intended for all (unless you're developing on this work by cloning it). Most use cases are covered in much lighter [demo folder](./../demo/).

## Table of contents

- [Scripts for exploring foundation models based VPR](#scripts-for-exploring-foundation-models-based-vpr)
    - [Table of contents](#table-of-contents)
    - [Contents](#contents)
        - [DINOv2](#dinov2)
        - [DINO](#dino)
        - [CLIP](#clip)
        - [CosPlace](#cosplace)
        - [Dataset Cluster Visualization](#dataset-cluster-visualization)
        - [Miscellaneous](#miscellaneous)
        - [Bash Scripts](#bash-scripts)

## Contents

Scripts that perform methods and tasks

### DINOv2

| S. No. | File | Description |
| :---- | :--- | :---- |
| 1 | [dino_v2_vlad.py](./dino_v2_vlad.py) | VLAD over DINOv2 ViT features |
| 2 | [dino_v2_gem.py](./dino_v2_gem.py) | GeM pooling over DINOv2 ViT features |
| 3 | [dino_v2_global_vpr.py](./dino_v2_global_vpr.py) | Use the CLS token of Dino-v2 as the global descriptor |
| 4 | [dino_v2_vlad_viz_layers.py](./dino_v2_vlad_viz_layers.py) | Visualize the DINOv2 Vocabulary for different ViT layers |
| 5 | [dino_v2_sim_facets.py](./dino_v2_sim_facets.py) | Similarity maps between/across ViT facets (in a layer) for query pixel (in a target image) using Dino-v2 features |
| 6 | [dino_v2_global_vocab_vlad.py](./dino_v2_global_vocab_vlad.py) | Generate VLAD cluster centers using collection of datasets (for domain-specific map/vocabulary) |
| 7 | [dino_v2_gp.py](./dino_v2_gp.py) | Pooling (max or average) DINOv2 ViT features |
| 8 | [dino_v2_vlad_global_vocab.py](./dino_v2_vlad_global_vocab.py) | IGNORE: VLAD over DINOv2 ViT features using a global vocabulary (concatenated all datasets). A special case of the [dino_v2_global_vocab_vlad.py](./dino_v2_global_vocab_vlad.py) script. |

### DINO

| S. No. | File | Description |
| :---- | :--- | :---- |
| 1 | [dino_vlad.py](./dino_vlad.py) | VLAD over DINO ViT extracted features |
| 2 | [dino_gem.py](./dino_gem.py) | GeM over DINO ViT extracted features |
| 3 | [dino_global_vpr.py](./dino_global_vpr.py) | VPR using CLS token |
| 4 | [dino_global_vocab_vlad.py](./dino_global_vocab_vlad.py) | VLAD over DINO ViT extracted features using custom (global map & domain-specific) vocabulary |
| 5 | [dino_vlad_viz_layers.py](./dino_vlad_viz_layers.py) | Visualize the DINO Vocabulary for different ViT layers |
| 6 | [dino_gp.py](./dino_gp.py) | Group pooling over DINO ViT features |
| 7 | [dino_vlad_contrastive_train.py](./dino_vlad_contrastive_train.py) | DEPRECATED: Contrastive training over Dino VLAD descriptors |
| 8 | [dino_multilayer_vlad.py](./dino_multilayer_vlad.py) | DEPRECATED: VLAD over multiple DINO ViT layer features |
| 9 | [dino_attnetion.py](./dino_attnetion.py) | DEPRECATED: Visualize Dino attentions |

### CLIP

| S. No. | File | Description |
| :---- | :--- | :---- |
| 1 | [clip_top_k_vpr.py](./clip_top_k_vpr.py) | CLIP features for VPR |
| 2 | [patch_clip.py](./patch_clip.py) | CLIP on patches of image -> VLAD |

### CosPlace

| S. No. | File | Description |
| :---- | :--- | :---- |
| 1 | [cosplace_vit_vlad.py](./cosplace_vit_vlad.py) | Extract features and do VLAD from CosPlace features (ViT backbone) |

### Dataset Cluster Visualization

For the cluster projections of the datasets

| S. No. | File | Description |
| :---- | :--- | :---- |
| 1 | [custom_gem_pca_clustering.py](./custom_gem_pca_clustering.py) | PCA clustering over GeM pooled descriptors of CosPlace, MixVPR, and NetVLAD |
| 2 | [custom_gem_tsne_clustering.py](./custom_gem_tsne_clustering.py) | t-SNE clustering over GeM pooled descriptors of CosPlace, MixVPR, and NetVLAD |
| 3 | [dino_v2_datasets_gem_pca_clustering.py](./dino_v2_datasets_gem_pca_clustering.py) | PCA clustering over GeM pooled DINOv2 ViT features |
| 4 | [dino_v2_datasets_gem_tsne_clustering.py](./dino_v2_datasets_gem_tsne_clustering.py) | t-SNE clustering over GeM pooled DINOv2 ViT features |
| 5 | [dino_v2_datasets_tsne_clustering.py](./dino_v2_datasets_tsne_clustering.py) | Dataset visualizations using t-SNE over DINOv2 ViT features (sub-sample all patches across all images) |

### Miscellaneous

Other scripts

| S. No. | File | Description |
| :----- | :--- | :---------- |
| 1 | [joint_pca_project.py](./joint_pca_project.py) | Project a set of global descriptors (from the same domain) "jointly" (see script `__doc__` for more) |
| 2 | [pca_downsample_experiment.py](./pca_downsample_experiment.py) | Load the downsampled points (most likely from [joint_pca_project.py](./joint_pca_project.py)) as global descriptors and run recall calculations. This script only runs recall calculations, taking global descriptors (everything else is logging for consistency). |

### Bash Scripts

For bulk runs

| S. No. | File | Description |
| :---- | :--- | :---- |
| 1 | [clip_ablations.sh](./clip_ablations.sh) | Run dataset ablations for CLIP |
| 2 | [cosplace_vit_ablations.sh](./cosplace_vit_ablations.sh) | Ablations for CosPlace ViT layer features (VLAD aggregation) |
| 3 | [dino_gem_ablations.sh](./dino_gem_ablations.sh) | DINO (v1) GeM ablations |
| 4 | [dino_global_ablations.sh](./dino_global_ablations.sh) | DINO (v1) global (CLS token) ablations |
| 5 | [dino_global_vocab_vlad_ablations.sh](./dino_global_vocab_vlad_ablations.sh) | DINOv2 VLAD (using global map & domain-specific vocabulary) ablations |
| 6 | [dino_gp_ablations.sh](./dino_gp_ablations.sh) | DINO (v1) group pooling (max and average) ablations |
| 7 | [dino_multilayer_vlad_ablations.sh](./dino_multilayer_vlad_ablations.sh) | DINO (v1) multi-layer (ViT feature) VLAD ablations |
| 8 | [dino_v2_gem_ablations.sh](./dino_v2_gem_ablations.sh) | DINOv2 (ViT features) GeM ablations |
| 9 | [dino_v2_global_ablations.sh](./dino_v2_global_ablations.sh) | DINOv2 global feature (CLS token) ablations |
| 10 | [dino_v2_global_vocab_vlad_ablations.sh](./dino_v2_global_vocab_vlad_ablations.sh) | Ablations over global vocabulary (cluster centers) made using DINOv2 VLAD (ViT features) |
| 11 | [dino_v2_gp_ablations.sh](./dino_v2_gp_ablations.sh) | Ablations for DINOv2 group pooling (max and average) over ViT features |
| 12 | [dino_v2_vlad_ablations.sh](./dino_v2_vlad_ablations.sh) | Ablations for DINOv2 VLAD (ViT features) |
| 13 | [dino_vlad_ablations.sh](./dino_vlad_ablations.sh) | DINO (v1) VLAD (ViT features) ablations |
| 14 | [pca_downsample_experiment_ablation.sh](./pca_downsample_experiment_ablation.sh) | PCA downsampling ablations. Experimental use only. |
