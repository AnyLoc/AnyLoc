# DINO v2 GeM Ablations
# 
# Usage: bash ./scripts/dino_v2_gem_ablations.sh
# 
#

# ---- Program arguments for user (after setting up datasets) ----
# Directory for storing experiment cache
cache_dir="/scratch/avneesh.mishra/vl-vpr/cache"
# Directory where the datasets are downloaded
data_vg_dir="/home2/avneesh.mishra/Documents/vl-vpr/datasets_vg/datasets"
# Dino models and layers: "Model, <Space separated layers>"
dino_models_layers=(
    "dinov2_vitg14, 31"
    # "dinov2_vitg14, `echo {39..0..1}`"
    # "dinov2_vitl14, `echo {23..0..1}`"
    # "dinov2_vitb14, `echo {11..0..1}`"
    # "dinov2_vits14, `echo {11..0..1}`"
    # "dinov2_vits14, 11"
)
# Facets
# dino_facets=("query" "key" "value" "token")
dino_facets=("value")
# GPU
gpu=${1:-0}
export CUDA_VISIBLE_DEVICES=$gpu
# Datasets
# datasets=("Oxford" "gardens" "17places" "baidu_datasets" "st_lucia" "pitts30k")
# datasets=("Oxford" "baidu_datasets")
# datasets=("VPAir")
datasets=("17places")
# WandB parameters
wandb_entity="vpr-vl"
# wandb_project="Paper_Dino-v2_Ablations"
wandb_project="Paper_Unstructured_Benchmarks"


# ----------- Main Experiment Code -----------
num_runs_dino=0
for ((i=0; i<${#dino_models_layers[*]}; i++)); do
    IFS="," read dino_model layers <<< "${dino_models_layers[$i]}"
    layers=(${layers})
    num_runs_dino=$(($num_runs_dino + ${#layers[@]}))
done
echo "Number of DINO runs: $num_runs_dino"
num_facets=${#dino_facets[@]}
echo "Num facets: $num_facets"
num_datasets=${#datasets[@]}
echo "Num datasets: $num_datasets"
total_runs=$(($num_runs_dino * $num_facets * $num_datasets))
curr_run=0
start_time=$(date)
start_time_secs=$SECONDS
echo "Start time: $start_time"
# For each dataset
for ((i=0; i<${#dino_models_layers[*]}; i++)); do
    IFS="," read dino_model layers <<< "${dino_models_layers[$i]}"
    layers=(${layers})
for layer in ${layers[*]}; do
for facet in ${dino_facets[*]}; do
for dataset in ${datasets[*]}; do
    # Header
    echo -ne "\e[1;93m"
    echo -n "--- => Model: $dino_model => Layer: $layer => Facet: $facet "
    echo -n "=> Dataset: $dataset "
    echo "---"
    curr_run=$((curr_run+1))
    echo "Run: $curr_run/$total_runs"
    echo -ne "\e[0m"
    # Variables for experiment
    wandb_group="${dataset}"
    wandb_name="DINO_V2_GeM/l${layer}_${facet}/${dataset}/${dino_model}"
    exp_id="ablations/$wandb_name"
    python_cmd="python ./scripts/dino_v2_gem.py"
    python_cmd+=" --exp-id $exp_id"
    python_cmd+=" --model-type ${dino_model}"
    python_cmd+=" --desc-layer $layer"
    python_cmd+=" --desc-facet $facet"
    python_cmd+=" --prog.cache-dir ${cache_dir}"
    python_cmd+=" --prog.data-vg-dir ${data_vg_dir}"
    python_cmd+=" --prog.vg-dataset-name ${dataset}"
    if [ "$dataset" == "pitts30k" ] || [ "$dataset" == "VPAir" ]; then
        # python_cmd+=" --sub-sample-db 10"
        python_cmd+=" --gem-elem-by-elem"
    fi
    # python_cmd+=" --prog.use-wandb"
    python_cmd+=" --prog.wandb-proj ${wandb_project}"
    python_cmd+=" --prog.wandb-entity ${wandb_entity}"
    python_cmd+=" --prog.wandb-group ${wandb_group}"
    python_cmd+=" --prog.wandb-run-name ${wandb_name}"
    echo -ne "\e[0;36m"
    echo $python_cmd
    echo -ne "\e[0m"
    # run_start_time=$(date)
    run_start_secs=$SECONDS
    $python_cmd
    # run_end_time=$(date)
    run_end_secs=$SECONDS
    # run_dur=$(echo $(date -d "$run_end_time" +%s) \
    #         - $(date -d "$run_start_time" +%s) | bc -l)
    run_dur=$(( $run_end_secs - $run_start_secs ))
    echo -n "---- Run finished in (HH:MM:SS): "
    echo "`date -d@$run_dur -u +%H:%M:%S` ----"
done
done
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

