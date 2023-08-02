# DINO VLAD Ablations
#
# Usage: bash ./scripts/dino_vlad_ablations.sh [GPU_NUM]
#
# GPU_NUM: 0, 1, 2, or 3
#   DINO facet is distributed on different GPUs
#
# 

# ---- Program arguments for user (after setting up datasets) ----
# Directory for storing experiment cache
cache_dir="/scratch/avneesh.mishra/vl-vpr/cache"
# Directory where the datasets are downloaded
data_vg_dir="/home2/avneesh.mishra/Documents/vl-vpr/datasets_vg/datasets"
# Dino models
dino_models=("dino_vits8")
# List of datasets
gpu=${1:-0}
# datasets=("pitts30k" "baidu_datasets" "gardens" "st_lucia" "Oxford" "17places")
dataset_settings=(
    "pitts30k, 11 6, key"
    "pitts30k, 11 8, key"
    "pitts30k, 11 8, value"
    "baidu_datasets, 11 6, key"
    "baidu_datasets, 10 11, value"
    "baidu_datasets, 9 8, value"
    "gardens, 11 6, key"
    "gardens, 8 7, key"
    "gardens, 8 7, query"
    "st_lucia, 11 6, key"
    "st_lucia, 6 7, query"
    "st_lucia, 6 7, key"
    "Oxford, 11 6, key"
    "Oxford, 8 5, query"
    "Oxford, 8 5, key"
    "17places, 11 6, key"
    "17places, 11 8, value"
    "17places, 11 5, value"
    "17places, 11 8 5, value"
    "17places, 11 8, query"
)
export CUDA_VISIBLE_DEVICES=$gpu
# Number of clusters for VLAD
num_clusters=(128)
# WandB parameters
wandb_entity="vpr-vl"
wandb_project="Ablations"
wandb_group="DINO_ML_VLAD"

# ----------- Main Experiment Code -----------
num_settings=${#dataset_settings[@]}
echo "Number of settings: $num_settings"
num_dino_models=${#dino_models[@]}
echo "Number of DINO models: $num_dino_models"
nu_clusters=${#num_clusters[@]}
echo "Number of clusters: $nu_clusters"
total_runs=$((
    num_settings * num_dino_models * nu_clusters
    ))
echo "Total number of runs: $total_runs"
curr_run=0
start_time=$(date)
echo "Start time: $start_time"
# For each dataset
for ((si=1; si<=$num_settings; si++)); do
    IFS="," read dataset dino_layer dino_facet <<< "${dataset_settings[$si-1]}"
    dino_layer=${dino_layer/ /}
    dino_facet=${dino_facet/ /}
for dino_model in ${dino_models[*]}; do
for num_cluster in ${num_clusters[*]}; do
    # Header
    echo -ne "\e[1;93m"
    echo -n "--- => Dataset: $dataset => DINO: $dino_model "
    echo -n "=> Clusters: $num_cluster => Layer: $dino_layer "
    echo "=> Facet: $dino_facet ---"
    curr_run=$((curr_run+1))
    echo "Run: $curr_run/$total_runs"
    echo -ne "\e[0m"
    # Variables for experiment
    wandb_name="DINO_ML_VLAD_l${dino_layer// /-}_c${num_cluster}/"
    wandb_name+="${dataset}/${dino_model}"
    exp_id="ablations/$wandb_name"
    python_cmd="python ./scripts/dino_multilayer_vlad.py"
    python_cmd+=" --exp-id $exp_id"
    python_cmd+=" --model-type ${dino_model}"
    python_cmd+=" --num-clusters ${num_cluster}"
    python_cmd+=" --desc-layers ${dino_layer}"
    python_cmd+=" --desc-facet ${dino_facet}"
    if [ "$dataset" == "pitts30k" ]; then
        python_cmd+=" --sub-sample-db-vlad 4"
    fi
    python_cmd+=" --prog.cache-dir ${cache_dir}"
    python_cmd+=" --prog.data-vg-dir ${data_vg_dir}"
    python_cmd+=" --prog.vg-dataset-name ${dataset}"
    python_cmd+=" --prog.wandb-entity ${wandb_entity}"
    python_cmd+=" --prog.wandb-proj ${wandb_project}"
    python_cmd+=" --prog.wandb-group ${wandb_group}"
    python_cmd+=" --prog.wandb-run-name ${wandb_name}"
    python_cmd+=" --no-cache-vlad-descs"    # Don't cache for now
    echo -ne "\e[0;36m"
    echo $python_cmd
    echo -ne "\e[0m"
    run_start_time=$(date)
    $python_cmd
    run_end_time=$(date)
    run_dur=$(echo $(date -d "$run_end_time" +%s) \
            - $(date -d "$run_start_time" +%s) | bc -l)
    echo -n "---- Run finished in (HH:MM:SS): "
    echo "`date -d@$run_dur -u +%H:%M:%S` ----"
done
done
done
end_time=$(date)
dur=$(echo $(date -d "$end_time" +%s) - $(date -d "$start_time" +%s) | bc -l)
_d=$(( dur/3600/24 ))
echo "---- Ablation took (d-HH:MM:SS): $_d-`date -d@$dur -u +%H:%M:%S` ----"
echo "Starting time: $start_time"
echo "Ending time: $end_time"
