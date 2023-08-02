# Perform CLIP + PCA ablations
# 
# Usage: bash ./scripts/clip_ablations.sh <gpu_id>
# 
# GPU ID: 1, 2, or 3
#   Different datasets are distributed on different GPUs
#   Assume that there are 4 GPUs, GPU 0 is reserved for other experiments


# ---- Program arguments for user (after setting up datasets) ----
# Directory for storing experiment cache
cache_dir="/scratch/avneesh.mishra/vl-vpr/cache"
# Directory where the datasets are downloaded
data_vg_dir="/home2/avneesh.mishra/Documents/vl-vpr/datasets_vg/datasets"
# Datasets to test
gpu=${1:-1}
if [ "$gpu" == "1" ]; then
    datasets=("st_lucia" "pitts30k")
elif [ "$gpu" == "2" ]; then
    datasets=("17places" "Oxford")
elif [ "$gpu" == "3" ]; then
    datasets=("gardens" "baidu_datasets")
else
    echo "Invalid GPU number: $gpu (1, 2, 3)"
    exit 1
fi
echo "Datasets: ${datasets[*]}"
export CUDA_VISIBLE_DEVICES=$gpu
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
# ---------------- Experiment parameters ----------------
# All the CLIP implementation to test
clip_impls=(    # CLIP Implementations: Impl, Backbone, Pretrained  # VRAM
    "openai, ViT-B/32, None"                        # < 6 GB
    "openai, RN50x64, None"                         # < 6 GB
    "openai, ViT-L/14@336px, None"                  # < 6 GB
    "open_clip, RN101-quickgelu, openai"            # < 6 GB
    "open_clip, coca_ViT-L-14, mscoco_finetuned_laion2b_s13b_b90k"  # < 6 GB
    "open_clip, ViT-g-14, laion2b_s12b_b42k"        # < 12 GB
    "open_clip, ViT-bigG-14, laion2b_s39b_b160k"    # < 16 GB
    "open_clip, convnext_large_d_320, laion2b_s29b_b131k_ft_soup"   # < 12 GB
    "open_clip, ViT-H-14, laion2b_s32b_b79k"        # < 12 GB
    # "open_clip, xlm-roberta-large-ViT-H-14, frozen_laion5b_s13b_b90k" # < 12 GB
)
# PCA paramters
pca_dims=(256 128 64 32)
pca_lowfs=(0.0 0.25 0.50 0.75 1.0)
# Weights and Biases
wandb_entity="vpr-vl"
wandb_group="CLIP_PCA"
wandb_project="Ablations"

# ----------- Main Experiment Code -----------
num_impls=${#clip_impls[@]}
echo "Number of CLIP implementations: $num_impls"
num_pca_dims=${#pca_dims[@]}
echo "Number of PCA dimensions: $num_pca_dims"
num_pca_lowfs=${#pca_lowfs[@]}
echo "Number of PCA low frequencies: $num_pca_lowfs"
curr_run=0
start_time=$(date)
# For each CLIP implementation
n=${#clip_impls[@]}
for ((i=0; i<$n; i++)); do
for pca_dim in ${pca_dims[*]}; do
for pca_lowf in ${pca_lowfs[*]}; do
    IFS=", " read clip_impl clip_backbone clip_pretrained <<< ${clip_impls[$i]}
    # For each dataset
    for dataset in ${datasets[*]}; do
        # Header
        echo -ne "\e[1;93m"
        echo -n "--- => CLIP: $clip_backbone ($clip_impl, $clip_pretrained)"
        echo " => Dataset: $dataset => PCA: $pca_dim, LF: $pca_lowf ---"
        curr_run=$((curr_run+1))
        echo "Run: $curr_run/$((num_impls*num_pca_dims*num_pca_lowfs))"
        echo -ne "\e[0m"
        # Variables for experiment
        wandb_name="CLIP_pca_D${pca_dim}_L${pca_lowf}/$dataset/$clip_backbone"
        python_cmd="python ./scripts/clip_top_k_vpr.py"
        python_cmd="$python_cmd --exp-id ablations/$wandb_name"
        python_cmd="$python_cmd --clip-impl $clip_impl"
        python_cmd="$python_cmd --clip-backbone $clip_backbone"
        python_cmd="$python_cmd --clip-pretrained $clip_pretrained"
        python_cmd="$python_cmd --faiss-method cosine"  # Default
        python_cmd="$python_cmd --pca-dim-reduce $pca_dim"
        python_cmd="$python_cmd --pca-low-factor $pca_lowf"
        python_cmd="$python_cmd --prog.cache-dir $cache_dir"
        python_cmd="$python_cmd --prog.data-vg-dir $data_vg_dir"
        python_cmd="$python_cmd --prog.vg-dataset-name $dataset"
        python_cmd="$python_cmd --prog.wandb-entity $wandb_entity"
        python_cmd="$python_cmd --prog.wandb-group $wandb_group"
        python_cmd="$python_cmd --prog.wandb-proj $wandb_project"
        python_cmd="$python_cmd --prog.wandb-run-name $wandb_name"
        echo -ne "\e[0;36m"
        echo $python_cmd
        echo -ne "\e[0m"
        run_start_time=$(date)
        $python_cmd
        run_end_time=$(date)
        run_dur=$(echo $(date -d "$run_end_time" +%s) \
                    - $(date -d "$run_start_time" +%s) | bc -l)
        echo -n "--- Run finished in (HH:MM:SS): "
        echo "`date -d@$run_dur -u +%H:%M:%S` ---"
    done
done
done
done
end_time=$(date)
dur=$(echo $(date -d "$end_time" +%s) - $(date -d "$start_time" +%s) | bc -l)
echo "--- Ablation took (HH:MM:SS): `date -d@$dur -u +%H:%M:%S` ---"

