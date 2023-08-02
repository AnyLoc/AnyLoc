# MAE VLAD Ablations
#
#
#
#

# --- Program arguments for user (after setting up datasets) ---
# Directory for storing experiment cache
cache_dir="/scratch/avneesh.mishra/vl-vpr/cache"
# Directory where the datasets are downloaded
data_vg_dir="/home2/avneesh.mishra/Documents/vl-vpr/datasets_vg/datasets"
# Models and type
ckpt_dir="./models/mae"
mae_models=(    # Checkpoint filename, Model class
    "mae_pretrain_vit_base.pth, mae_vit_base_patch16",
    "mae_pretrain_vit_large.pth, mae_vit_large_patch16",
    "mae_pretrain_vit_huge.pth, mae_vit_huge_patch14",
    "mae_visualize_vit_large.pth, mae_vit_large_patch16",
    "mae_visualize_vit_large_ganloss.pth, mae_vit_large_patch16",
)
# Datasets
datasets=("gardens" "Oxford" "17places" "baidu_datasets" "st_lucia" "pitts30k")
# Number of clusters for VLAD
num_clusters=(32 64 128 256)
# WandB parameters
wandb_entity="vpr-vl"
wandb_project="Ablations"
wandb_group="MAE_VLAD_patch"


num_models=${#mae_models[@]}
num_datasets=${#datasets[@]}
nu_clusters=${#num_clusters[@]}
echo "Number of models: $num_models"
echo "Number of datasets: $num_datasets"
echo "Number of clusters: $nu_clusters"
total_runs=$((num_models * num_datasets * nu_clusters))
echo "Total number of runs: $total_runs"
curr_run=0
start_time=$(date)
echo "Start time: $start_time"
for ((i=0; i<num_models; i++)); do
for dataset in ${datasets[*]}; do
for nc in ${num_clusters[*]}; do
    IFS=', ' read ckpt model_class <<< "${mae_models[$i]}"
    # Header
    echo -ne "\e[1;93m"
    echo "-----------------------------------------------"
    echo -ne "=> Model checkpoint: $ckpt => Class: $model_class\n"
    echo -ne "=> Dataset: $dataset \n"
    echo -ne "=> Number of clusters: $nc \n"
    curr_run=$((curr_run+1))
    echo "Run: $curr_run/$total_runs"
    echo "-----------------------------------------------"
    echo -ne "\e[0m"
    # Variables for experiment
    wandb_name="MAE_VLAD_c${nc}/${model_class}/${dataset}"
    exp_id="ablations/${wandb_name}"
    python_cmd="python ./scripts/mae_vlad.py"
    python_cmd+=" --exp-id ${exp_id}"
    python_cmd+=" --ckpt-path ${ckpt_dir}/${ckpt}"
    python_cmd+=" --mae-model ${model_class}"
    python_cmd+=" --num-clusters ${nc}"
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
    $python_cmd
    run_end_time=$(date)
    run_dur=$(echo $(date -d "$run_end_time" +%s) \
            - $(date -d "$run_start_time" +%s) | bc -l)
    echo -n "---- Run finished in (HH:MM:SS): "
    echo "`date -d@$run_dur -u +%H:%M:%S` ----"
done
done
done
end_time=$(date)
dur=$(echo $(date -d "$end_time" +%s) - $(date -d "$start_time" +%s) | bc -l)
_d=$(( dur/3600/24 ))
echo "---- Ablation took (d-HH:MM:SS): $_d-`date -d@$dur -u +%H:%M:%S` ----"
echo "Starting time: $start_time"
echo "Ending time: $end_time"
