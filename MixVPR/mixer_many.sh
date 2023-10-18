# Run MixVPR
#
# Usage:
#   bash ./mixer_many.sh
#
# Run this script from inside the `MixVPR` folder
#   - 
#

# Backup from Jay
# python3 -u mixer_top_k_vpr.py --prog.data-vg-dir /ocean/projects/cis220039p/shared/datasets/vpr/datasets_vg --prog.vg-dataset-name laurel_caverns --prog.wandb_proj Unstructured --prog.wandb_entity vpr-vl --prog.wandb_group laurel_caverns --prog.wandb_run_name MixVPR/laurel_caverns
# python3 -u mixer_top_k_vpr.py --prog.data-vg-dir /ocean/projects/cis220039p/shared/datasets/vpr/datasets_vg --prog.vg-dataset-name hawkins_long_corridor --prog.wandb_proj Unstructured --prog.wandb_entity vpr-vl --prog.wandb_group hawkins_long_corridor --prog.wandb_run_name MixVPR/hawkins_long_corridor
# python3 -u mixer_top_k_vpr.py --prog.data-vg-dir /ocean/projects/cis220039p/shared/datasets/vpr/datasets_vg --prog.vg-dataset-name VPAir --prog.wandb_proj Unstructured --prog.wandb_entity vpr-vl --prog.wandb_group VPAir --prog.wandb_run_name MixVPR/VPAir
# python3 -u mixer_top_k_vpr.py --prog.data-vg-dir /ocean/projects/cis220039p/shared/datasets/vpr/datasets_vg --prog.vg-dataset-name eiffel #--prog.wandb_proj Unstructured --prog.wandb_entity vpr-vl --prog.wandb_group eiffel --prog.wandb_run_name MixVPR/eiffel

# python ./mixer_top_k_vpr.py --prog.cache-dir /scratch/avneesh.mishra/vl-vpr/cache --prog.data-vg-dir /home2/avneesh.mishra/Documents/vl-vpr/datasets_vg/datasets --prog.vg-dataset-name Oxford --ckpt-file "/home2/avneesh.mishra/Documents/vl-vpr/MixVPR/resnet50_MixVPR_4096_channels(1024)_rows(4) (1).ckpt" --qual-result-percent 0.0

# Directory for storing experiment cache (not used)
cache_dir="/scratch/avneesh.mishra/vl-vpr/cache"
cache_dir+="/MixVPR_runs"
# Directory where the checkpoints are stored
ckpts_dir="/home2/avneesh.mishra/Documents/vl-vpr/models/MixVPR"
# ckpt_file="$ckpts_dir/resnet50_MixVPR_4096_channels(1024)_rows(4) (1).ckpt"
ckpt_file="$ckpts_dir/resnet50_default.ckpt"    # Don't put spaces here
ckpt_wandb_name="ResNet50_MixVPR_4096_c1024"
# Directory where the datasets are downloaded
# data_vg_dir="/ocean/projects/cis220039p/shared/datasets/vpr/datasets_vg"
data_vg_dir="/home2/avneesh.mishra/Documents/vl-vpr/datasets_vg/datasets"
# Datasets
gpu=${1:-0}
export CUDA_VISIBLE_DEVICES=$gpu
# datasets=("Oxford" "gardens" "17places" "baidu_datasets" "st_lucia" "pitts30k")
# datasets=("Tartan_GNSS_rotated" "Tartan_GNSS_notrotated" "hawkins" "laurel_caverns")
# datasets=("Tartan_GNSS_test_rotated" "Tartan_GNSS_test_notrotated")
# datasets=("eiffel")
# datasets=("Tartan_GNSS_test_rotated" "Tartan_GNSS_test_notrotated" "Tartan_GNSS_rotated" "Tartan_GNSS_notrotated" "hawkins" "laurel_caverns" "eiffel" "VPAir")
datasets=("Oxford_25m")
# WandB parameters
wandb_entity="vpr-vl"
# wandb_project="Paper_Structured_Benchmarks"
# wandb_project="Paper_Unstructured_Benchmarks"
wandb_project="Rebuttal_Experiments"

num_datasets=${#datasets[@]}
total_runs=$(( num_datasets ))
echo "Total number of runs: $total_runs"
curr_run=0
start_time=$(date)
start_time_secs=$SECONDS
echo "Start time: $start_time"
for dataset in ${datasets[*]}; do
    # Header
    echo -ne "\e[1;93m"
    echo "--- => Dataset: $dataset ---"
    curr_run=$((curr_run+1))
    echo "Run: $curr_run/$total_runs"
    echo -ne "\e[0m"
    wandb_group="${dataset}"
    wandb_name="MixVPR/${dataset}/$ckpt_wandb_name"
    python_cmd="python ./mixer_top_k_vpr.py"
    python_cmd+=" --prog.cache-dir $cache_dir"
    python_cmd+=" --prog.data-vg-dir $data_vg_dir"
    python_cmd+=" --prog.vg-dataset-name $dataset"
    python_cmd+=" --ckpt-file $ckpt_file"
    python_cmd+=" --qual-result-percent 0.0"
    # python_cmd+=" --prog.use-wandb"
    python_cmd+=" --prog.wandb-entity $wandb_entity"
    python_cmd+=" --prog.wandb-proj $wandb_project"
    python_cmd+=" --prog.wandb-group $wandb_group"
    python_cmd+=" --prog.wandb-run-name $wandb_name"
    # python_cmd+=" --save-all-descs dataset_clusters/MixVPR/${dataset}"
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
end_time=$(date)
end_time_secs=$SECONDS
# dur=$(echo $(date -d "$end_time" +%s) - $(date -d "$start_time" +%s) | bc -l)
dur=$(( $end_time_secs - $start_time_secs ))
_d=$(( dur/3600/24 ))
echo "---- Ablation took (d-HH:MM:SS): $_d-`date -d@$dur -u +%H:%M:%S` ----"
echo "Starting time: $start_time"
echo "Ending time: $end_time"
