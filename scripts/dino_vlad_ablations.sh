# DINO VLAD Ablations
#
# Usage: bash ./scripts/dino_vlad_ablations.sh [GPU_NUM]
# 
# 

# ---- Program arguments for user (after setting up datasets) ----
# Directory for storing experiment cache
cache_dir="/scratch/avneesh.mishra/vl-vpr/cache"
# Directory where the datasets are downloaded
data_vg_dir="/home2/avneesh.mishra/Documents/vl-vpr/datasets_vg/datasets"
# Dino models
dino_models=("dino_vits8" "dino_vits16" "dino_vitb8" "dino_vitb16")
# List of Dino Layers to inspect
dino_layers=(`echo {11..0}`)
# List of datasets
gpu=${1:-0}
export CUDA_VISIBLE_DEVICES=$gpu
# datasets=("Oxford" "gardens" "17places" "baidu_datasets" "st_lucia" "pitts30k")
datasets=("Oxford" "baidu_datasets")
dino_facets=("key" "query" "value" "token")
# Number of clusters for VLAD
num_clusters=(128)
# WandB parameters
wandb_entity="vpr-vl"
wandb_project="Paper_Dino_Ablations"

# ----------- Main Experiment Code -----------
num_datasets=${#datasets[@]}
echo "Number of datasets: $num_datasets"
num_dino_models=${#dino_models[@]}
echo "Number of DINO models: $num_dino_models"
nu_clusters=${#num_clusters[@]}
echo "Number of clusters: $nu_clusters"
num_layers=${#dino_layers[@]}
echo "Number of DINO layers: $num_layers"
num_facets=${#dino_facets[@]}
echo "Number of DINO facets: $num_facets"
total_runs=$((
    num_datasets * num_dino_models * nu_clusters * num_layers * num_facets
    ))
echo "Total number of runs: $total_runs"
curr_run=0
start_time=$(date)
start_time_secs=$SECONDS
echo "Start time: $start_time"
# For each dataset
for dataset in ${datasets[*]}; do
for dino_model in ${dino_models[*]}; do
for num_cluster in ${num_clusters[*]}; do
for dino_layer in ${dino_layers[*]}; do
for dino_facet in ${dino_facets[*]}; do
    # Header
    echo -ne "\e[1;93m"
    echo -n "--- => Dataset: $dataset => DINO: $dino_model "
    echo -n "=> Clusters: $num_cluster => Layer: $dino_layer "
    echo "=> Facet: $dino_facet ---"
    curr_run=$((curr_run+1))
    echo "Run: $curr_run/$total_runs"
    echo -ne "\e[0m"
    # Variables for experiment
    wandb_group="$dataset"
    wandb_name="DINO_VLAD/l${dino_layer}_${dino_facet}_c${num_cluster}/"
    wandb_name+="${dataset}/${dino_model}"
    exp_id="ablations/$wandb_name"
    python_cmd="python ./scripts/dino_vlad.py"
    python_cmd+=" --exp-id $exp_id"
    python_cmd+=" --model-type ${dino_model}"
    python_cmd+=" --num-clusters ${num_cluster}"
    python_cmd+=" --desc-layer ${dino_layer}"
    python_cmd+=" --desc-facet ${dino_facet}"
    if [ "$dataset" == "pitts30k" ]; then
        python_cmd+=" --sub-sample-db-vlad 4"
    fi
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
done
end_time=$(date)
end_time_secs=$SECONDS
# dur=$(echo $(date -d "$end_time" +%s) - $(date -d "$start_time" +%s) | bc -l)
dur=$(( $end_time_secs - $start_time_secs ))
_d=$(( dur/3600/24 ))
echo "---- Ablation took (d-HH:MM:SS): $_d-`date -d@$dur -u +%H:%M:%S` ----"
echo "Starting time: $start_time"
echo "Ending time: $end_time"
