# Runs with Dino-v2 using global (mixed) vocabulary
#
# Usage: bash ./scripts/dino_v2_global_vocab_ablations.sh
#

# ---- Program arguments for user (after setting up datasets) ----
# Directory for storing VLAD cache (an ID will be used in a subfolder)
cache_dir="/scratch/avneesh.mishra/vl-vpr/cache/video_clusters"
# Directory where the datasets are downloaded
data_vg_dir="/home2/avneesh.mishra/Documents/vl-vpr/datasets_vg/datasets"
# Dino models and layers: "Model, <Space separated layers>"
dino_models_layers=(
    "dinov2_vitg14, 31"
    # "dinov2_vitg14, `echo {39..0..1}`"
    # "dinov2_vitl14, `echo {23..0..1}`"
    # "dinov2_vitb14, `echo {11..0..1}`"
    # "dinov2_vits14, `echo {11..0..1}`"
    # "dinov2_vits14, 10"
)
# Facets
dino_facets=("value")
# Datasets
# datasets=("eiffel")
datasets=("VPAir")
# Number of VLAD clusters
# num_clusters=(256 128 64 32)
num_clusters=(32)
# Global vocabulary (datasets to include)
# global_vocabs=("structured" "unstructured" "both")
# global_vocabs=("both")
# global_vocabs=("indoor" "urban" "aerial" "hawkins" "laurel_caverns")
global_vocabs=("aerial")
# GPU
gpu=${1:-0}
export CUDA_VISIBLE_DEVICES=$gpu
# WandB parameters
# wandb_entity="vpr-vl"
# wandb_project="Ablations"
# wandb_group="DINO_V2_VLAD"

wandb_entity="vpr-vl"
# wandb_project="Paper_Dino-v2_Ablations"
wandb_project="Paper_Dino-v2_Global_Vocab_Ablations"


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
nu_clusters=${#num_clusters[@]}
echo "Num clusters: $nu_clusters"
nu_gvoc=${#global_vocabs[@]}
echo "Num global vocabs: $nu_gvoc"
total_runs=$(( num_runs_dino * num_facets * num_datasets * nu_clusters * nu_gvoc ))
echo "Total runs: $total_runs"
curr_run=0
start_time=$(date)
start_time_secs=$SECONDS
echo "Start time: $start_time"
for ((i=0; i<${#dino_models_layers[*]}; i++)); do
    IFS="," read dino_model layers <<< "${dino_models_layers[$i]}"
    layers=(${layers})
for layer in ${layers[*]}; do
for nc in ${num_clusters[*]}; do
for facet in ${dino_facets[*]}; do
for dataset in ${datasets[*]}; do
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
    wandb_name="DINO_V2_VLAD_GLOBAL_VOCAB/${model_id}/${dataset}/${dino_model}"
    exp_id="exp/6b82"
    python_cmd="python ./scripts/dino_v2_global_vocab_vlad.py"
    python_cmd+=" --exp-id ${exp_id}"
    python_cmd+=" --vlad-cache-dir ${cache_dir}/vpr_global_vocab/${dino_model}/${model_id}"
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
    elif [ "$global_vocab" == "hawkins" ]; then
        python_cmd+=" --db-samples.hawkins 1"
    elif [ "$global_vocab" == "laurel_caverns" ]; then
        python_cmd+=" --db-samples.laurel-caverns 1"
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
