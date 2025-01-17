#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
which python
export LD_LIBRARY_PATH=/opt/conda/envs/llava/lib/python3.10/site-packages/nvidia/nvjitlink/lib:/usr/local/cuda/lib64:/usr/local/cuda/compat/lib.real:$LD_LIBRARY_PATH

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python playground/calvin_data_audio/prepare_audios.py \
        --instructions ./playground/calvin_data_audio/task_ABCD_D.jsonl \
        --calvin_audio_dir /zhaowei/data/calvin_audio \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX &
done

wait