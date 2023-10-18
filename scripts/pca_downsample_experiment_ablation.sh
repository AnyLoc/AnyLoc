# PCA downsampling experiment
#
# Usage: bash ./scripts/pca_downsample_experiment_ablation.sh
#

# ---------- Program arguments
# Cache directory
cache_dir="/scratch/$USER/vl-vpr/cache"
# Directory where datasets are stored
data_vg_dir="/home2/avneesh.mishra/Documents/vl-vpr/datasets_vg/datasets"

# WandB properties
wandb_proj="Rebuttal_Experiments-PCA"
wandb_group="Domain-JointReduce"

# Run parameters
pca_dims=(1024 512 256 128 64 32 16)
ds_names=("baidu_datasets" "gardens" "17places" "pitts30k" "st_lucia" "Oxford_25m")
model_type=("dino_v2")

# Main runs
num_pca_dims=${#pca_dims[@]}
echo "Number of PCA dims: ${num_pca_dims}"
num_ds_names=${#ds_names[@]}
echo "Number of datasets: ${num_ds_names}"
num_model_types=${#model_type[@]}
echo "Number of models: ${num_model_types}"
total_runs=$((num_pca_dims * num_ds_names * num_model_types))
echo "Total runs: ${total_runs}"
curr_run=0
start_time=$(date)
start_time_secs=$SECONDS
echo "Start time: $start_time"
for pca_dim in ${pca_dims[@]}; do
for ds_name in ${ds_names[@]}; do
for model in ${model_type[@]}; do
    # Header
    echo -ne "\e[1;93m"
    echo -n "--- => PCA Dim $pca_dim => Dataset $ds_name => Model: $model "
    echo "---"
    curr_run=$((curr_run+1))
    echo "Run: $curr_run/$total_runs"
    echo -ne "\e[0m"
    # Parameters
    if [ "$model" == "dino" ]; then
        exp_id="pca_dr_dino_v1/$pca_dim"
        base_dir="${cache_dir}/experiments/pca_downsample_dino_v1"
        model_arch="dino_vits8"
        desc_layer=9
        desc_facet="key"
        num_c=128
        wb_name="DINO_VLAD/"
    elif [ "$model" == "dino_v2" ]; then
        exp_id="pca_dr/$pca_dim"
        base_dir="${cache_dir}/experiments/pca_downsample"
        model_arch="dinov2_vitg14"
        desc_layer=31
        desc_facet="value"
        num_c=32
        wb_name="DINO_V2_VLAD/"
    else
        echo "Invalid model: $model"
        exit 1
    fi
    wb_name+="l${desc_layer}_${desc_facet}_c${num_c}/$ds_name/$model_arch"
    # Run
    python_cmd="python ./scripts/pca_downsample_experiment.py"
    python_cmd+=" --prog.use-wandb"
    python_cmd+=" --prog.wandb-proj $wandb_proj"
    python_cmd+=" --prog.wandb-group $wandb_group"
    python_cmd+=" --prog.wandb-run-name $wb_name"
    python_cmd+=" --prog.cache-dir $cache_dir"
    python_cmd+=" --prog.data-vg-dir $data_vg_dir"
    python_cmd+=" --prog.vg-dataset-name $ds_name"
    python_cmd+=" --base-dir $base_dir"
    python_cmd+=" --exp-id $exp_id"
    python_cmd+=" --pca-dim-reduce $pca_dim"
    python_cmd+=" --model-type $model_arch"
    python_cmd+=" --desc-layer $desc_layer"
    python_cmd+=" --desc-facet $desc_facet"
    python_cmd+=" --num-clusters $num_c"
    echo -ne "\e[0;36m"
    echo $python_cmd
    echo -ne "\e[0m"
    # run_start_time=$(date)
    run_start_secs=$SECONDS
    $python_cmd
    # # Delay (because it's too fast and we want some time b/w caching)
    # sleep 1
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
end_time=$(date)
end_time_secs=$SECONDS
# dur=$(echo $(date -d "$end_time" +%s) - $(date -d "$start_time" +%s) | bc -l)
dur=$(( $end_time_secs - $start_time_secs ))
_d=$(( dur/3600/24 ))
echo "---- Ablation took (d-HH:MM:SS): $_d-`date -d@$dur -u +%H:%M:%S` ----"
echo "Starting time: $start_time"
echo "Ending time: $end_time"
