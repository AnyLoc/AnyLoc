#!/bin/bash

# This script is used to generate the facet ablation plots for the paper.

python3 facet_ablation_plot.py -d 'Baidu Mall' 'Oxford' \
    -i ./data/ablations/facet/dinov2_baidu.csv ./data/ablations/facet/dinov2_oxford.csv \
    -m 'DINOv2 ViT-G14 L31' \
    -o /ocean/projects/cis220039p/nkeetha/data/vlm/found/ablation_plots/facet/DINOv2_ViT_G14.pdf

python3 facet_ablation_plot.py -d 'Baidu Mall' 'Pitts-30k' \
    -i ./data/ablations/facet/dinov2_baidu.csv ./data/ablations/facet/dinov2_pitt.csv \
    -m 'DINOv2 ViT-G14 L31' \
    -o /ocean/projects/cis220039p/nkeetha/data/vlm/found/ablation_plots/facet/DINOv2_ViT_G14_Pitt.pdf

python3 facet_ablation_plot.py -d 'Baidu Mall' 'Oxford' \
    -i ./data/ablations/facet/dino_l9_baidu.csv ./data/ablations/facet/dino_l9_oxford.csv \
    -m 'DINO ViT-S8 L9' \
    -o /ocean/projects/cis220039p/nkeetha/data/vlm/found/ablation_plots/facet/DINO_ViT_S8_L9.pdf

python3 facet_ablation_plot.py -d 'Baidu Mall' 'Pitts-30k' \
    -i ./data/ablations/facet/dino_l9_baidu.csv ./data/ablations/facet/dino_l9_pitt.csv \
    -m 'DINO ViT-S8 L9' \
    -o /ocean/projects/cis220039p/nkeetha/data/vlm/found/ablation_plots/facet/DINO_ViT_S8_L9_Pitt.pdf