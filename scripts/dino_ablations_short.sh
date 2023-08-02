# A Short Dino ablation study
#
# Usage: bash ./scripts/dino_ablations_short.sh
#
# Each GPU runs a set of datasets
#

# Program parameters
dino_models=( "dino_vits8" "dino_vits16" "dino_vitb8" "dino_vitb16" "vit_small_patch16_224" "vit_base_patch8_224" "vit_base_patch16_224" )
num_clusters=128
data_vg_dir="/home2/avneesh.mishra/Documents/vl-vpr/datasets_vg/datasets"
cache_dir="/scratch/avneesh.mishra/vl-vpr/cache"
# Datasets to test
gpu=${1:-0}
if [ $gpu -eq 0 ]; then
    # datasets=( "baidu_datasets" "pitts30k" "17places" )
    datasets=( "pitts30k" )
    dino_layer=11
elif [ $gpu -eq 1 ]; then
    datasets=( "st_lucia" "Oxford" "gardens" )
    dino_layer=6
else
    echo "GPU $gpu not supported (should be 0 or 1)"
    exit 1
fi
export CUDA_VISIBLE_DEVICES=$gpu
# Weights and Biases
wandb_entity="vpr-vl"
wandb_project="DINO-ViT-Ablations"

# ----------- Main Experiment Code -----------
num_datasets=${#datasets[@]}
echo "Number of datasets: $num_datasets"
num_models=${#dino_models[@]}
echo "Number of DINO models: $num_models"
curr_run=0
start_time=$(date)
# For each dataset
for ((i=0; i<$num_datasets; i++)); do
    dataset=${datasets[$i]}
    # For each model
    for ((j=0; j<$num_models; j++)); do
        dino_model=${dino_models[$j]}
        # Header
        echo -ne "\e[1;93m"
        echo "--- => Dataset: $dataset => Model: $dino_model ---"
        curr_run=$((curr_run+1))
        echo -e "Run: $curr_run / $((num_datasets*num_models))"
        echo -ne "\e[0m"
        # Variables for experiment
        wandb_name="$dataset/${dino_model}_hard_l${dino_layer}_c${num_clusters}"
        python_cmd="python ./scripts/dino_vlad.py"
        python_cmd+=" --exp-id DINO-ViT-Ablations/${wandb_name}"
        python_cmd+=" --model-type $dino_model"
        python_cmd+=" --num-clusters $num_clusters"
        python_cmd+=" --desc-layer $dino_layer"
        python_cmd+=" --desc-facet key"
        python_cmd+=" --prog.cache-dir $cache_dir"
        python_cmd+=" --prog.data-vg-dir $data_vg_dir"
        python_cmd+=" --prog.vg-dataset-name $dataset"
        python_cmd+=" --prog.wandb-proj $wandb_project"
        python_cmd+=" --prog.wandb-entity $wandb_entity"
        python_cmd+=" --prog.wandb-group $dataset"
        python_cmd+=" --prog.wandb-run-name $wandb_name"
        if [ "$dataset" == "pitts30k" ]; then
            python_cmd+=" --sub-sample-db-vlad 2"
        fi
        echo -ne "\e[0;36m"
        echo $python_cmd
        echo -ne "\e[0m"
        run_start_time=$(date)
        $python_cmd
        run_end_time=$(date)
        run_dur=$(echo $(date -d "$run_end_time" +%s) \
                    - $(date -d "$run_start_time" +%s) | bc -l)
        echo -n "--- Run finished in (HH:MM:SS): "
        echo "`date -d@$run_dur -u +%H:%M:%S` ---"
    done
done
end_time=$(date)
dur=$(echo $(date -d "$end_time" +%s) - $(date -d "$start_time" +%s) | bc -l)
echo "--- Ablation took (HH:MM:SS): `date -d@$dur -u +%H:%M:%S` ---"
