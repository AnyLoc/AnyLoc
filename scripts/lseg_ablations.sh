# Run LSeg work
#
# Usage: bash ./scripts/lseg_ablations_env.sh [ID]
#
# ID: 0 or 1 for splitting clusters across GPUs
#   Note that GPU doesn't matter here (all through CPU as of now)

# ---- User program arguments (after setting up datasets and cache) ----
# Directory for storing experiment cache
cache_dir="/scratch/avneesh.mishra/vl-vpr/cache"
# Directory where the datasets are downloaded
data_vg_dir="/home2/avneesh.mishra/Documents/vl-vpr/datasets_vg/datasets"
# -- LSeg configurations --
datasets_vg_dir="/scratch/avneesh.mishra/lseg/datasets_vg_cache"
# Database directories for each dataset
declare -A db_dirs
db_dirs["Oxford"]="$datasets_vg_dir/Oxford_Robotcar/oxDataPart/1-s-resized"
db_dirs["gardens"]="$datasets_vg_dir/gardens/day_right"
db_dirs["st_lucia"]="$datasets_vg_dir/st_lucia/test/database"
db_dirs["17places"]="$datasets_vg_dir/17places/ref"
db_dirs["pitts30k"]="$datasets_vg_dir/pitts30k/test/database"
db_dirs["baidu_datasets"]="$datasets_vg_dir/baidu_datasets/training_images_undistort"
# Query directories for each dataset
declare -A qu_dirs
qu_dirs["Oxford"]="$datasets_vg_dir/Oxford_Robotcar/oxDataPart/2-s-resized"
qu_dirs["gardens"]="$datasets_vg_dir/gardens/night_right"
qu_dirs["st_lucia"]="$datasets_vg_dir/st_lucia/test/queries"
qu_dirs["17places"]="$datasets_vg_dir/17places/query"
qu_dirs["pitts30k"]="$datasets_vg_dir/pitts30k/test/queries"
qu_dirs["baidu_datasets"]="$datasets_vg_dir/baidu_datasets/query_images_undistort"
# ----- Ablation parameters -----
# Datasets
# datasets=("Oxford" "gardens" "st_lucia" "17places" "pitts30k" "baidu_datasets")
datasets=("pitts30k" "baidu_datasets")
# Number of clusters
gpu=${1:-0}
num_clusters=(32 64 128 256)
# if [ "$gpu" -eq 0 ]; then
#     num_clusters=(32 64)
# elif [ "$gpu" -eq 1 ]; then
#     num_clusters=(256 128)
# else
#     echo "Invalid identifier"
#     exit 1
# fi
export CUDA_VISIBLE_DEVICES=$gpu
# Wandb parameters
wandb_entity="vpr-vl"
wandb_project="Ablations"
wandb_group="LSeg-VLAD"

# ----------- Main Experiment Code -----------
num_datasets=${#datasets[@]}
echo "Number of datasets: $num_datasets"
nu_clusters=${#num_clusters[@]}
echo "Number of clusters: $nu_clusters"
total_runs=$((num_datasets * nu_clusters))
echo "Total number of runs: $total_runs"
curr_run=0
start_time=$(date)
echo "Start time: $start_time"
for dataset in "${datasets[@]}"; do
for num_cluster in "${num_clusters[@]}"; do
    # Header
    echo -ne "\e[1;93m"
    echo -n "--- => Dataset $dataset => Clusters: $num_cluster ---"
    curr_run=$((curr_run+1))
    echo "Run: $curr_run/$total_runs"
    echo -ne "\e[0m"
    if [ $curr_run -lt 3 ]; then
        echo "Skipping"
        continue
    fi
    # Variables for experiment
    wandb_name="${wandb_group}_c${num_cluster}/${dataset}"
    exp_id="ablations/${wandb_name}"
    python_cmd="python ./scripts/lseg_vlad.py"
    python_cmd+=" --exp-id ${exp_id}"
    python_cmd+=" --query-cache-dir ${qu_dirs[$dataset]}"
    python_cmd+=" --db-cache-dir ${db_dirs[$dataset]}"
    python_cmd+=" --num-clusters ${num_cluster}"
    python_cmd+=" --sub-sample-db-vlad 2"
    python_cmd+=" --prog.cache-dir ${cache_dir}"
    python_cmd+=" --prog.data-vg-dir ${data_vg_dir}"
    python_cmd+=" --prog.vg-dataset-name ${dataset}"
    python_cmd+=" --prog.wandb-proj ${wandb_project}"
    python_cmd+=" --prog.wandb-entity ${wandb_entity}"
    python_cmd+=" --prog.wandb-group ${wandb_group}"
    python_cmd+=" --prog.wandb-run-name ${wandb_name}"
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
end_time=$(date)
dur=$(echo $(date -d "$end_time" +%s) - $(date -d "$start_time" +%s) | bc -l)
_d=$(( dur/3600/24 ))
echo "---- Ablation took (d-HH:MM:SS): $_d-`date -d@$dur -u +%H:%M:%S` ----"
echo "Starting time: $start_time"
echo "Ending time: $end_time"

