# CosPlace ViT Layer Ablations
#
# Usage: bash ./scripts/cosplace_vit_ablations.sh
#
#

# ---- Program arguments for user (after setting up datasets) ----
# Directory for storing experiment cache
cache_dir="/scratch/avneesh.mishra/vl-vpr/cache"
# Directory where the datasets are downloaded
data_vg_dir="/home2/avneesh.mishra/Documents/vl-vpr/datasets_vg/datasets"
# Checkpoint
ckpt_dir="/scratch/avneesh.mishra/vl-vpr/cache/.models/CosPlace"
ckpt_file="${ckpt_dir}/vit_best_model.pth"
# Layers
vit_layers=(`echo {11..0}`)
# Facets
vit_facets=("key" "query" "value" "token")
# List of datasets
gpu=${1:-0}
export CUDA_VISIBLE_DEVICES=$gpu
# datasets=("Oxford" "gardens" "17places" "baidu_datasets" "st_lucia" "pitts30k")
if [ "$HOSTNAME" == "gnode085" ]; then
    datasets=("hawkins")
    # Number of cluster
    if [ "$gpu" -eq 1 ]; then
        num_clusters=(128)
    elif [ "$gpu" -eq 2 ]; then
        num_clusters=(64)
    elif [ "$gpu" -eq 3 ]; then
        num_clusters=(32)
    else
        echo "Invalid GPU number"
        exit 1
    fi
elif [ "$HOSTNAME" == "gnode117" ]; then
    datasets=("baidu_datasets")
    # Number of cluster
    if [ "$gpu" -eq 0 ]; then
        num_clusters=(128)
    elif [ "$gpu" -eq 1 ]; then
        num_clusters=(64)
    elif [ "$gpu" -eq 2 ]; then
        num_clusters=(32)
    else
        echo "Invalid GPU number"
        exit 1
    fi
else
    echo "Invalid host: $HOSTNAME"
    exit 1
fi
# WandB parameters
wandb_entity="vpr-vl"
wandb_project="Ablations"
wandb_group="CosPlace_ViT_VLAD"


# ----------- Main Experiment Code -----------
num_datasets=${#datasets[@]}
echo "Number of datasets: $num_datasets"
num_layers=${#vit_layers[@]}
echo "Number of layers: $num_layers"
num_facets=${#vit_facets[@]}
echo "Number of facets: $num_facets"
num_c=${#num_clusters[@]}
total_runs=$((
    num_datasets * num_layers * num_facets * num_c
))
curr_run=0
start_time=$(date)
start_time_secs=$SECONDS
echo "Start time: $start_time"
for dataset in "${datasets[@]}"; do
for layer in "${vit_layers[@]}"; do
for facet in "${vit_facets[@]}"; do
for nc in "${num_clusters[@]}"; do
    # Header
    echo -ne "\e[1;93m"
    echo -n "--- => Dataset: $dataset => Layer: $layer => Facet: $facet "
    echo "=> Num Clusters: $nc ---"
    curr_run=$((curr_run+1))
    echo "Run: $curr_run/$total_runs"
    echo -ne "\e[0m"
    # Variables for experiment
    wandb_name="CosPlace_VLAD/l${layer}_${facet}_c${nc}"
    wandb_name+="/${dataset}"
    exp_id="ablations/${wandb_name}"
    python_cmd="python ./scripts/cosplace_vit_vlad.py"
    python_cmd+=" --exp-id ${exp_id}"
    python_cmd+=" --vit-ckpt-path ${ckpt_file}"
    python_cmd+=" --num-clusters ${nc}"
    python_cmd+=" --desc-layer ${layer}"
    python_cmd+=" --desc-facet ${facet}"
    python_cmd+=" --prog.cache-dir ${cache_dir}"
    python_cmd+=" --prog.data-vg-dir ${data_vg_dir}"
    python_cmd+=" --prog.vg-dataset-name ${dataset}"
    python_cmd+=" --prog.use-wandb"
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

