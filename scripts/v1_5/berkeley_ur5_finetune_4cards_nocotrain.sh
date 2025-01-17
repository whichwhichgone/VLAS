#!/bin/bash
which python
echo $PATH
export LD_LIBRARY_PATH=/opt/conda/envs/llava/lib/python3.10/site-packages/nvidia/nvjitlink/lib:/usr/local/cuda/lib64:/usr/local/cuda/compat/lib.real:$LD_LIBRARY_PATH
export WANDB_API_KEY=83793606f810aa3d385ea5d12dbd352514ac54e1
export WANDB_MODE=offline


deepspeed llava/train/berkeley_ur5.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./checkpoints/llava-v1.5-7b-audio-connector \
    --version v1 \
    --data_path ./playground/berkeley_ur5_audio/berkeley_autolab_ur5_training_audio.json \
    --image_folder /storage/zhaowei/data/berkeley_autolab_ur5/vla_processed \
    --vision_tower /zhaowei/models/clip-vit-large-patch14-336 \
    --audio_folder /storage/zhaowei/data/ur5_audio/berkeley_autolab_ur5 \
    --audio_tower /zhaowei/models/models--openai--whisper-large-v2/snapshots/ae4642769ce2ad8fc292556ccea8e901f1530655 \
    --action_stat /storage/zhaowei/data/berkeley_our_ur5/training/statistics.yaml \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir /storage/zhaowei/checkpoints/llava-v1.5-7b-ur5-reduce5-cotrain-4cards-nocotrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy no \
    --save_strategy epoch \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
