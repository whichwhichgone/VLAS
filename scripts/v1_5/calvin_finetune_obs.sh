#!/bin/bash
export WANDB_API_KEY=83793606f810aa3d385ea5d12dbd352514ac54e1

deepspeed llava/train/calvin_train_obs.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path /zhaowei/models/llava-v1.5-7b \
    --version v1 \
    --data_path ./playground/calvin_data/task_ABCD_D_training_r5.json \
    --image_folder ./playground/calvin_data/task_ABCD_D \
    --action_stat /wangdonglin/calvin/task_ABCD_D/training/statistics.yaml \
    --vision_tower /zhaowei/models/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-calvin-rel-obs-reduce5-v2-abcd2d \
    --num_train_epochs 2 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy no \
    --save_strategy epoch \
    --save_total_limit 2 \
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
    --report_to_wandb_project vlas \
    --report_to_wandb_run_name vlas_ft_stage3_text_baseline_abcd2d
