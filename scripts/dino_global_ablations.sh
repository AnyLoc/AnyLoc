# DINO (v1) global (CLS of token facet from last layer) Ablations
#
# Usage: bash ./scripts/dino_global_ablations.sh [GPU_NUM]
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
export CUDA_VISIBLE_DEVICES=$gpu
# datasets=("Oxford" "gardens" "17places" "baidu_datasets" "st_lucia" "pitts30k")
# datasets=("Oxford" "baidu_datasets")
# datasets=("VPAir")
# datasets=("Tartan_GNSS_rotated" "Tartan_GNSS_notrotated" "Tartan_GNSS_test_rotated" "Tartan_GNSS_test_notrotated" "hawkins" "laurel_caverns" "eiffel" "VPAir")  # Use only "test" for Tartan
datasets=("Oxford_25m")
# WandB parameters
wandb_entity="vpr-vl"
# wandb_project="Paper_Structured_Benchmarks"
# wandb_project="Paper_Unstructured_Benchmarks"
wandb_project="Rebuttal_Experiments"


# ----------- Main Experiment Code -----------
num_datasets=${#datasets[@]}
echo "Number of datasets: $num_datasets"
num_dino_models=${#dino_models[@]}
echo "Number of DINO models: $num_dino_models"
total_runs=$(( num_datasets * num_dino_models ))
echo "Total number of runs: $total_runs"
curr_run=0
start_time=$(date)
start_time_secs=$SECONDS
echo "Start time: $start_time"
# For each dataset
for dataset in ${datasets[*]}; do
for dino_model in ${dino_models[*]}; do
    # Header
    echo -ne "\e[1;93m"
    echo -n "--- => Dataset: $dataset => DINO: $dino_model --- "
    curr_run=$((curr_run+1))
    echo "Run: $curr_run/$total_runs"
    echo -ne "\e[0m"
    # Variables for experiment
    wandb_group="${dataset}"
    wandb_name="DINO_GLOBAL/${dataset}/${dino_model}"
    exp_id="ablations/$wandb_name"
    python_cmd="python ./scripts/dino_global_vpr.py"
    python_cmd+=" --exp-id $exp_id"
    python_cmd+=" --model-type ${dino_model}"
    python_cmd+=" --prog.cache-dir ${cache_dir}"
    python_cmd+=" --prog.data-vg-dir ${data_vg_dir}"
    python_cmd+=" --prog.vg-dataset-name ${dataset}"
    # python_cmd+=" --prog.use-wandb"
    python_cmd+=" --prog.wandb-entity ${wandb_entity}"
    python_cmd+=" --prog.wandb-proj ${wandb_project}"
    python_cmd+=" --prog.wandb-group ${wandb_group}"
    python_cmd+=" --prog.wandb-run-name ${wandb_name}"
    echo -ne "\e[0;36m"
    echo $python_cmd
    echo -ne "\e[0m"
    run_start_time=$(date)
    echo "- Run start time: ${run_start_time} -"
    run_start_secs=$SECONDS
    $python_cmd
    run_end_time=$(date)
    echo "- Run end time: ${run_end_time} -"
    run_end_secs=$SECONDS
    # run_dur=$(echo $(date -d "$run_end_time" +%s) \
    #         - $(date -d "$run_start_time" +%s) | bc -l)
    run_dur=$(( $run_end_secs - $run_start_secs ))
    echo -n "---- Run finished in (HH:MM:SS): "
    echo "`date -d@$run_dur -u +%H:%M:%S` ----"
done
done
end_time=$(date)
end_time_secs=$SECONDS
# dur=$(echo $(date -d "$end_time" +%s) - $(date -d "$start_time" +%s) | bc -l)
dur=$(( $end_time_secs - $start_time_secs ))
_d=$(( dur/3600/24 ))
echo "---- Ablation took (d-HH:MM:SS): $_d-`date -d@$dur -u +%H:%M:%S` ----"
echo "Starting time: $start_time"
echo "Ending time: $end_time"
