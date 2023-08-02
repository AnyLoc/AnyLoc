# ImageBind as global descriptor
#
# Usage bash ./scripts/imagebind_global_ablations.sh
#
#

# ---- Program arguments for user (after setting up datasets) ----
# Directory for storing experiment cache
cache_dir="/scratch/avneesh.mishra/vl-vpr/cache"
# Directory where the datasets are downloaded
data_vg_dir="/home2/avneesh.mishra/Documents/vl-vpr/datasets_vg/datasets"
# Checkpoint file
ckpt_file="./models/imagebind/imagebind_huge.pth"
model_name="imagebind_huge"
# Datasets
gpu=${1:-0} # GPU
if [ $gpu -eq 1 ]; then
    datasets=("st_lucia" "pitts30k")
elif [ $gpu -eq 2 ]; then
    datasets=("17places" "baidu_datasets")
elif [ $gpu -eq 3 ]; then
    datasets=("Oxford" "gardens")
else
    echo "Invalid GPU number (use 1, 2, or 3)"
    exit 1
fi
export CUDA_VISIBLE_DEVICES=$gpu
# WandB parameters
wandb_entity="vpr-vl"
wandb_project="Ablations"
wandb_group="ImageBind_global"

# ----------- Main Experiment Code -----------
num_datasets=${#datasets[@]}
echo "Number of datasets: $num_datasets"
total_runs=$((num_datasets))
echo "Total number of runs: $total_runs"
curr_run=0
start_time=$(date)
echo "Start time: $start_time"
# For each dataset
for dataset in ${datasets[*]}; do
    # Header
    echo -ne "\e[1;93m"
    echo "--- => Dataset: $dataset ---"
    curr_run=$((curr_run+1))
    echo "Run: $curr_run/$total_runs"
    echo -ne "\e[0m"
    # Variables for experiment
    wandb_name="${wandb_group}/r224/${model_name}/${dataset}"
    exp_id="ablations/${wandb_name}"
    python_cmd="python ./scripts/imagebind_global_vpr.py"
    python_cmd+=" --exp-id ${exp_id}"
    python_cmd+=" --model-ckpt-path ${ckpt_file}"
    python_cmd+=" --prog.cache-dir ${cache_dir}"
    python_cmd+=" --prog.data-vg-dir ${data_vg_dir}"
    python_cmd+=" --prog.vg-dataset-name ${dataset}"
    python_cmd+=" --prog.use-wandb"
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
end_time=$(date)
dur=$(echo $(date -d "$end_time" +%s) - $(date -d "$start_time" +%s) | bc -l)
_d=$(( dur/3600/24 ))
echo "---- Ablation took (d-HH:MM:SS): $_d-`date -d@$dur -u +%H:%M:%S` ----"
echo "Starting time: $start_time"
echo "Ending time: $end_time"

