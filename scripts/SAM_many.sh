# --------------------- Paths ---------------------
# Cache directory (where images and model cache will be stored)
cache_dir="/ocean/projects/cis220039p/jkarhade/data/sam_cache" #"/home/jay/Documents/vl-vpr/cache" 
# Dataset path (where the datasets are downloaded)
ds_path="/ocean/projects/cis220039p/shared/datasets/vpr/datasets_vg" #"/home/jay/Downloads/vl_vpr_datasets" #"/home2/avneesh.mishra/Documents/vl-vpr/datasets_vg/datasets"
# Model path (where the model vit for SAM is downloaded)

# ----------------- General parameters -----------------
# The datasets downloaded (for testing)
datasets=("pitts30k")
exp_prefix="SAM_Ablation"
# SAM parameters
model_types=("vit_b")
out_layer_numbers=(12)

# VLAD parameters
num_clusters=(128)
vlad_mode="hard"
vlad_sm_temp="1.0"

# ----------------- Weights and Biases parameters -----------------
wandb_project="SAM_Ablation"     # Project name (set to "None" for no WandB)

export CUDA_VISIBLE_DEVICES=0

for model_type in ${model_types[*]}; do
for num_cluster in ${num_clusters[*]}; do
for out_layer_num in ${out_layer_numbers[*]}; do
for ds in ${datasets[*]}; do
    exp_id=$(head /dev/urandom | md5sum -b | awk '{print substr($1,0,4)}')
    echo -ne "\e[1;93m"
    echo -e " => Dataset: $ds \n => Num clusters: $num_cluster"
    echo -ne "\e[0m"
    if [[ ! -z $exp_prefix ]]; then
        exp_id="$exp_prefix/$exp_id"
    fi
    exp_cache_dir="$cache_dir/experiments/$exp_id"
    while [[ -d $exp_id ]]; do
        echo "Experiment folder already exists, trying a new one"
        exp_id=$(head /dev/urandom | md5sum -b | awk '{print substr($1,0,4)}')
        if [[ ! -z $exp_prefix ]]; then
            exp_id="$exp_prefix/$exp_id"
        fi
    done
    # Construct the python command

    wandb_group="$ds"
    wandb_run_name="$ds/$num_cluster/$model_type"

    python_cmd="python ./SAM_vlad.py"
    python_cmd="$python_cmd --exp-id $exp_id --model-type $model_type"
    python_cmd="$python_cmd --num-clusters $num_cluster"
    python_cmd="$python_cmd --out_layer_num $out_layer_num"
    python_cmd="$python_cmd --prog.cache-dir $cache_dir"
    python_cmd="$python_cmd --prog.data-vg-dir $ds_path"
    python_cmd="$python_cmd --prog.vg-dataset-name $ds"
    python_cmd="$python_cmd --vlad-assignment $vlad_mode"
    python_cmd="$python_cmd --vlad-soft-temp $vlad_sm_temp"
    if [[ "$ds" == "pitts30k" ]]; then  # FIXME: For running on 128 GB RAM
        python_cmd="$python_cmd --sub-sample-db-vlad 4"
    fi
    if [[ $wandb_project == "None" ]]; then
        python_cmd="$python_cmd --prog.no-use-wandb"
    else
        python_cmd="$python_cmd --prog.wandb-proj $wandb_project"
        python_cmd="$python_cmd --prog.wandb-group $wandb_group"
        python_cmd="$python_cmd --prog.wandb-run-name $wandb_run_name"        
    fi
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo -ne "\e[0;36m"
    echo $python_cmd
    echo -ne "\e[0m"
    eval $python_cmd    # Main execution
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
done
done
done
done
done
done