#!/bin/bash
which python
export LD_LIBRARY_PATH=/opt/conda/envs/llava/lib/python3.10/site-packages/nvidia/nvjitlink/lib:/usr/local/cuda/lib64:/usr/local/cuda/compat/lib.real:$LD_LIBRARY_PATH
export WANDB_API_KEY=local-73e66a04fa97e2a5d5c573a97e65bf1194533e1f
export WANDB_BASE_URL=http://10.28.0.22:30437/

deepspeed llava/train/asr_train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /zhaowei/models/llava-v1.5-7b \
    --version plain \
    --data_path ./playground/data/LLaVA-Pretrain/librispeech_asr_train_clean_100.json \
    --image_folder ./playground/data/LLaVA-Pretrain/images \
    --vision_tower /zhaowei/models/clip-vit-large-patch14-336 \
    --audio_folder /zhaowei/data/LLaVA-Audio \
    --audio_folder_asr /zhaowei/data/LibriSpeech/train-clean-100 \
    --audio_tower /zhaowei/models/models--openai--whisper-large-v2/snapshots/ae4642769ce2ad8fc292556ccea8e901f1530655 \
    --mm_projector_type mlp2x_gelu \
    --tune_audio_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-pretrain-audio-connector-v1 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
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
    --report_to_wandb_run_name vlas_ft_stage1
