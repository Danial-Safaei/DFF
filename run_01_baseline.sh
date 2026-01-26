#!/usr/bin/env bash
set -euo pipefail

accelerate launch /networkhome/WMGDS/safaei_d/3_generators/training/vkitti_seg_final7/ControlNet/train_controlnet_sdxl_vkitti_baseline_20251221_185155.py \
  --train_data_dir /networkhome/WMGDS/safaei_d/3_generators/training/vkitti_seg/controlnet_kitti_to_vkitti_dataset \
  --output_dir /networkhome/WMGDS/safaei_d/3_generators/training/vkitti_seg_final7/out_01_baseline \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --controlnet_model_name_or_path SargeZT/sdxl-controlnet-seg \
  --image_column target_image --conditioning_image_column conditioning_image --caption_column prompt \
  --resolution 1248x376 \
  --train_batch_size 4 --gradient_accumulation_steps 1 \
  --max_train_steps 60000 \
  --learning_rate 2e-05 --lr_scheduler constant_with_warmup --lr_warmup_steps 250 \
  --mixed_precision bf16 --report_to tensorboard \
  --set_grads_to_none --dataloader_num_workers 8 --allow_tf32 \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps 999999999
