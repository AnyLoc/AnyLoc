#!/bin/bash

# This script is used to generate the layer ablation plots for the paper.

python3 layer_ablation_plot.py -d 'Baidu Mall' 'Oxford' \
    -i ./data/ablations/vit_and_layer/dinov2_G14_baidu.csv ./data/ablations/vit_and_layer/dinov2_G14_oxford.csv \
    -m 'DINOv2 ViT-G14 Value' \
    -o /ocean/projects/cis220039p/nkeetha/data/vlm/found/ablation_plots/layer/DINOv2_ViT_G14.pdf

python3 layer_ablation_plot.py -d 'Baidu Mall' 'Pitts-30k' \
    -i ./data/ablations/vit_and_layer/dinov2_G14_baidu.csv ./data/ablations/vit_and_layer/dinov2_G14_pitt.csv \
    -m 'DINOv2 ViT-G14 Value' \
    -o /ocean/projects/cis220039p/nkeetha/data/vlm/found/ablation_plots/layer/DINOv2_ViT_G14_Pitt.pdf

# python3 layer_ablation_plot.py -d 'Baidu Mall' 'Oxford' \
#     -i ./data/ablations/vit_and_layer/dinov2_L14_baidu.csv ./data/ablations/vit_and_layer/dinov2_L14_oxford.csv \
#     -m 'DINOv2 ViT-L14 Value' \
#     -o /ocean/projects/cis220039p/nkeetha/data/vlm/found/ablation_plots/layer/DINOv2_ViT_L14.pdf

# python3 layer_ablation_plot.py -d 'Baidu Mall' 'Oxford' \
#     -i ./data/ablations/vit_and_layer/dinov2_B14_baidu.csv ./data/ablations/vit_and_layer/dinov2_B14_oxford.csv \
#     -m 'DINOv2 ViT-B14 Value' \
#     -o /ocean/projects/cis220039p/nkeetha/data/vlm/found/ablation_plots/layer/DINOv2_ViT_B14.pdf

# python3 layer_ablation_plot.py -d 'Baidu Mall' 'Oxford' \
#     -i ./data/ablations/vit_and_layer/dinov2_S14_baidu.csv ./data/ablations/vit_and_layer/dinov2_S14_oxford.csv \
#     -m 'DINOv2 ViT-S14 Value' \
#     -o /ocean/projects/cis220039p/nkeetha/data/vlm/found/ablation_plots/layer/DINOv2_ViT_S14.pdf

python3 layer_ablation_plot.py -d 'Baidu Mall' 'Oxford' \
    -i ./data/ablations/vit_and_layer/dino_key_S8_baidu.csv ./data/ablations/vit_and_layer/dino_key_S8_oxford.csv \
    -m 'DINO ViT-S8 Key' \
    -o /ocean/projects/cis220039p/nkeetha/data/vlm/found/ablation_plots/layer/DINO_ViT_S8_Key.pdf

python3 layer_ablation_plot.py -d 'Baidu Mall' 'Pitts-30k' \
    -i ./data/ablations/vit_and_layer/dino_key_S8_baidu.csv ./data/ablations/vit_and_layer/dino_key_S8_pitt.csv \
    -m 'DINO ViT-S8 Key' \
    -o /ocean/projects/cis220039p/nkeetha/data/vlm/found/ablation_plots/layer/DINO_ViT_S8_Key_Pitt.pdf

# python3 layer_ablation_plot.py -d 'Baidu Mall' 'Oxford' \
#     -i ./data/ablations/vit_and_layer/dino_value_S8_baidu.csv ./data/ablations/vit_and_layer/dino_value_S8_oxford.csv \
#     -m 'DINO ViT-S8 Value' \
#     -o /ocean/projects/cis220039p/nkeetha/data/vlm/found/ablation_plots/layer/DINO_ViT_S8_Value.pdf