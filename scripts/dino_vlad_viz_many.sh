# Run many Dino visualization scripts
#
# Usage: bash ./scripts/dino_vlad_viz_many.sh
#

# ---- Program arguments for user (after setting up datasets) ----
# Directory for storing experiment cache
cache_dir="/scratch/avneesh.mishra/vl-vpr/cache"
# Directory where the datasets are downloaded
data_vg_dir="/home2/avneesh.mishra/Documents/vl-vpr/datasets_vg/datasets"
# Datasets to test (dataset, sub_sample_qu) pair
gpu=${1:-1}
if [ "$gpu" == "1" ]; then
    datasets=("17places, 10" "pitts30k, 100")
elif [ "$gpu" == "2" ]; then
    datasets=("Oxford, 5" "st_lucia, 30")
elif [ "$gpu" == "3" ]; then
    datasets=("baidu_datasets, 55" "gardens, 5")
else
    echo "Invalid GPU number: $gpu"
    echo "Valid GPU numbers: 1, 2, or 3"
    exit 1
fi
export CUDA_VISIBLE_DEVICES=$gpu
echo "GPU: $CUDA_VISIBLE_DEVICES"
# Number of clusters to test
nu_clusters=(4 16 128)
# Weights and Biases
wandb_proj="Dino-Descs"
wandb_group="viz_vlad_clusters"
# Dino parameters
dino_model="dino_vits8"
dino_layer=11
dino_facet="key"

# ---------- Main Expeirment Code -----------
num_datasets=${#datasets[@]}
echo "Number of datasets: $num_datasets"
num_clusters=${#nu_clusters[@]}
echo "Number of clusters: $num_clusters"
total_runs=$((
    num_datasets * num_clusters
    ))
echo "Total number of runs: $total_runs"
curr_run=0
start_time=$(date)
echo "Start time: $start_time"
# For each dataset > cluster
for ((i=0; i<$num_datasets; i++)); do
    IFS=", " read dataset sub_sample_qu <<< ${datasets[$i]}
for nc in ${nu_clusters[*]}; do
    # Header
    echo -ne "\e[1;93m"
    echo "--- => Dataset: $dataset => S_QU: $sub_sample_qu => Clusters: $nc ---"
    curr_run=$((curr_run+1))
    echo "Run: $curr_run/$total_runs"
    echo -ne "\e[0m"
    # Variables for experiment
    wandb_name="$dataset/$nc/$dino_model"
    exp_id="dino_vlad_viz/L${dino_layer}-${dino_facet}/$wandb_name"
    python_cmd="python ./scripts/dino_vlad_viz.py"
    python_cmd+=" --exp-id $exp_id"
    python_cmd+=" --model-type ${dino_model}"
    python_cmd+=" --num-clusters ${nc}"
    if [ "$dataset" == "pitts30k" ]; then
        python_cmd+=" --sub-sample-db-vlad 4"
    fi
    python_cmd+=" --sub-sample-qu ${sub_sample_qu}"
    python_cmd+=" --desc-layer ${dino_layer}"
    python_cmd+=" --desc-facet ${dino_facet}"
    python_cmd+=" --prog.cache-dir ${cache_dir}"
    python_cmd+=" --prog.data-vg-dir ${data_vg_dir}"
    python_cmd+=" --prog.vg-dataset-name ${dataset}"
    python_cmd+=" --prog.wandb-proj ${wandb_proj}"
    python_cmd+=" --prog.wandb-group ${wandb_group}"
    python_cmd+=" --prog.wandb-run-name ${wandb_name}"
    # Run the experiment
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
