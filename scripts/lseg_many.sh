#!/usr/bin/env bash
# Run the LSeg experiment in bulk

echo "Python: $(which python)"
exp_id=$(head /dev/urandom | md5sum -b | awk '{print substr($1,0,4)}')
echo "Experiment ID: $exp_id"
read -p "Confirm (Ctrl-c to exit)? "

export CUDA_VISIBLE_DEVICES=2

echo "CUDA: $CUDA_VISIBLE_DEVICES"

try_nc=(32 64 128)
for nc in ${try_nc[*]}; do

    # if (( $nc >= 128 )); then
    # # Pitts30k
    #     echo "::::::::: Dataset: pitts30k, | Clusters = $nc (with inorm) :::::::::"
    #     python ./scripts/lseg_vlad.py --qurey-cache-dir /scratch/avneesh.mishra/lseg/datasets_vg_cache/pitts30k/test/queries --db-cache-dir /scratch/avneesh.mishra/lseg/datasets_vg_cache/pitts30k/test/database --prog.vg-dataset-name pitts30k --sub-sampling-db 6 --sub-sampling-qu 6 --num-clusters $nc --exp-id "p30_${nc}_${exp_id}" --vlad-assignment soft
    # else
    #     echo "::::::::: Dataset: pitts30k, Clusters = $nc (with inorm) :::::::::"
    #     python ./scripts/lseg_vlad.py --qurey-cache-dir /scratch/avneesh.mishra/lseg/datasets_vg_cache/pitts30k/test/queries --db-cache-dir /scratch/avneesh.mishra/lseg/datasets_vg_cache/pitts30k/test/database --prog.vg-dataset-name pitts30k --sub-sampling-db 2 --sub-sampling-qu 2 --num-clusters $nc --exp-id "p30_${nc}_${exp_id}" --vlad-assignment soft
    # fi

    # St. Lucia
    # echo "::::::::: Dataset: st_lucia, Clusters = $nc (with inorm) :::::::::"
    # python ./scripts/lseg_vlad.py --qurey-cache-dir /scratch/avneesh.mishra/lseg/datasets_vg_cache/st_lucia/test/queries --db-cache-dir /scratch/avneesh.mishra/lseg/datasets_vg_cache/st_lucia/test/database --prog.vg-dataset-name st_lucia --sub-sampling-db 2 --sub-sampling-qu 2 --num-clusters $nc --exp-id "stl_${nc}_${exp_id}" --vlad-assignment soft

    # 17places
    echo "::::::::: Dataset: 17places, Clusters = $nc (hard) :::::::::"
    python ./scripts/lseg_vlad.py --qurey-cache-dir /scratch/avneesh.mishra/lseg/datasets_vg_cache/17places/query --db-cache-dir /scratch/avneesh.mishra/lseg/datasets_vg_cache/17places/ref --gt-npy-file /scratch/avneesh.mishra/vl-vpr/data/17places/ground_truth_new.npy --db-type vpr_bench --prog.vg-dataset-name 17places --sub-sampling-db 2 --sub-sampling-qu 2 --num-clusters $nc --exp-id "17p_${nc}_${exp_id}" --vlad-assignment hard

    # 17places
    echo "::::::::: Dataset: 17places, Clusters = $nc (soft) :::::::::"
    python ./scripts/lseg_vlad.py --qurey-cache-dir /scratch/avneesh.mishra/lseg/datasets_vg_cache/17places/query --db-cache-dir /scratch/avneesh.mishra/lseg/datasets_vg_cache/17places/ref --gt-npy-file /scratch/avneesh.mishra/vl-vpr/data/17places/ground_truth_new.npy --db-type vpr_bench --prog.vg-dataset-name 17places --sub-sampling-db 2 --sub-sampling-qu 2 --num-clusters $nc --exp-id "17p_${nc}_${exp_id}" --vlad-assignment soft
done
