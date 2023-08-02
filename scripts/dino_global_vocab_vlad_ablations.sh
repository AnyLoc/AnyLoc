# Runs with Dino-v2 using global (mixed) vocabulary
#
# Usage: bash ./scripts/dino_v2_global_vocab_ablations.sh
#

# ---- Program arguments for user (after setting up datasets) ----
# Directory for storing VLAD cache (an ID will be used in a subfolder)
cache_dir="/scratch/avneesh.mishra/vl-vpr/cache"
# Directory where the datasets are downloaded
data_vg_dir="/home2/avneesh.mishra/Documents/vl-vpr/datasets_vg/datasets"
# Dino models
dino_models=("dino_vits8")
dino_layers=(9)
dino_facets=("key")
# Number of clusters for VLAD
num_clusters=(128)
# List of datasets
gpu=${1:-0}
export CUDA_VISIBLE_DEVICES=$gpu
# datasets=("Oxford" "gardens" "17places" "baidu_datasets" "st_lucia" "pitts30k")
datasets=("Oxford")
# Global vocabulary (datasets to include)
# global_vocabs=("structured" "unstructured" "both")
global_vocabs=("indoor" "urban" "aerial")
# WandB parameters
wandb_entity="vpr-vl"
wandb_project="Paper_Dino-v1_Global_Vocab_Ablations"

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
for nc in ${num_clusters[*]}; do
for layer in ${dino_layers[*]}; do
for facet in ${dino_facets[*]}; do
for global_vocab in ${global_vocabs[*]}; do
    # Header
    echo -ne "\e[1;93m"
    echo -n "--- => Model: $dino_model => Layer: $layer => Num Clusters: $nc "
    echo -ne "\n    => Facet: $facet => Dataset: $dataset => Global Vocab: $global_vocab "
    echo "---"
    curr_run=$((curr_run+1))
    echo "Run: $curr_run/$total_runs"
    echo -ne "\e[0m"
    # Variables for experiment
    wandb_group="${global_vocab}"
    # ID for VLAD cache (model identifier)
    model_id="l${layer}_${facet}_c${nc}/${global_vocab}"
    wandb_name="DINO_V1_VLAD_GLOBAL_VOCAB/${model_id}/${dataset}/${dino_model}"
    exp_id="ablations/$wandb_name"
    python_cmd="python ./scripts/dino_global_vocab_vlad.py"
    python_cmd+=" --exp-id ${exp_id}"
    python_cmd+=" --vlad-cache-dir ${cache_dir}/vpr_global_dino_v1/${model_id}"
    python_cmd+=" --num-clusters $nc"
    python_cmd+=" --model-type ${dino_model}"
    python_cmd+=" --desc-layer $layer"
    python_cmd+=" --desc-facet $facet"
    python_cmd+=" --prog.data-vg-dir ${data_vg_dir}"
    python_cmd+=" --prog.cache-dir ${cache_dir}"
    python_cmd+=" --prog.vg-dataset-name ${dataset}"
    # python_cmd+=" --prog.use-wandb"
    python_cmd+=" --prog.wandb-proj ${wandb_project}"
    python_cmd+=" --prog.wandb-entity ${wandb_entity}"
    python_cmd+=" --prog.wandb-group ${wandb_group}"
    python_cmd+=" --prog.wandb-run-name ${wandb_name}"
    if [ "$global_vocab" == "indoor" ]; then
        python_cmd+=" --db-samples.baidu-datasets 1"
        python_cmd+=" --db-samples.gardens 1"
        python_cmd+=" --db-samples.17places 1"
    elif [ "$global_vocab" == "urban" ]; then
        python_cmd+=" --db-samples.Oxford 1"
        python_cmd+=" --db-samples.st-lucia 1"
        python_cmd+=" --db-samples.pitts30k 4"
    elif [ "$global_vocab" == "aerial" ]; then
        python_cmd+=" --db-samples.Tartan-GNSS-test-rotated 1"
        python_cmd+=" --db-samples.Tartan-GNSS-test-notrotated 1"
        python_cmd+=" --db-samples.VPAir 2"
    else
        echo "Invalid global vocab!"
        exit 1
    fi
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
