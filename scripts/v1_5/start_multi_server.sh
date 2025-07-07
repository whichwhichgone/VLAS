#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
which python
export LD_LIBRARY_PATH=/opt/conda/envs/llava/lib/python3.10/site-packages/nvidia/nvjitlink/lib:/usr/local/cuda/lib64:/usr/local/cuda/compat/lib.real:$LD_LIBRARY_PATH

# python -m llava.serve.flask_helper &

for IDX in $(seq 0 $((CHUNKS-1))); do
    port=$(($IDX+9002))
    echo "Running port $port on GPU ${GPULIST[$IDX]}"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.serve.flask_server \
        --model-path /zhaowei/workspace/VLAS/checkpoints/llava-v1.5-7b-calvin-rel-obs-reduce5-audio-v2-abcd2d/checkpoint-5393 \
        --action_stat /wangdonglin/calvin/task_ABCD_D/training/statistics.yaml \
        --port $port &
done

wait