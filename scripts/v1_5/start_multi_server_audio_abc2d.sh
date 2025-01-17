#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
which python
export LD_LIBRARY_PATH=/opt/conda/envs/llava/lib/python3.10/site-packages/nvidia/nvjitlink/lib:/usr/local/cuda/lib64:/usr/local/cuda/compat/lib.real:$LD_LIBRARY_PATH

python -m llava.serve.flask_helper &

for IDX in $(seq 0 $((CHUNKS-1))); do
    port=$(($IDX+9002))
    echo "Running port $port on GPU ${GPULIST[$IDX]}"
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.serve.flask_server_audio_abc2d \
        --model-path  /zhaowei/workspace/LLaVA/checkpoints/llava-v1.5-7b-calvin-rel-obs-reduce5-audio-v1-abc2d \
        --action_stat /wangdonglin/calvin/task_ABC_D/training/statistics.yaml \
        --port $port \
        --eval_instructions /zhaowei/workspace/LLaVA/playground/calvin_data_audio/eval/new_custom_validation.yaml \
        --eval_audio /zhaowei/data/calvin_eval_audio_custom &
done

wait
