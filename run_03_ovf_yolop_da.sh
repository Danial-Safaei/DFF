#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256

accelerate launch /networkhome/WMGDS/safaei_d/3_generators/training/vkitti_seg_final7/ControlNet/train_controlnet_sdxl_vkitti_ovf_yolop_da_corrected_20251222_001624.py \
  --train_data_dir /networkhome/WMGDS/safaei_d/3_generators/training/vkitti_seg_final7/controlnet_vkitti_ovf_yolop_da_corrected_dataset \
  --output_dir /networkhome/WMGDS/safaei_d/3_generators/training/vkitti_seg_final7/out_03_ovf_yolop_da_beta0p50 \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --controlnet_model_name_or_path SargeZT/sdxl-controlnet-seg \
  --image_column image --conditioning_image_column conditioning_image --caption_column prompt \
  --resolution 1248x376 \
  --train_batch_size 1 --gradient_accumulation_steps 2 \
  --gradient_checkpointing \
  --max_train_steps 60000 \
  --learning_rate 2e-05 --lr_scheduler constant_with_warmup --lr_warmup_steps 250 \
  --mixed_precision bf16 --report_to tensorboard \
  --set_grads_to_none --dataloader_num_workers 8 --allow_tf32 \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps 999999999 \
  --yolop_root /networkhome/WMGDS/safaei_d/3_generators/models/pretrained/YOLOP \
  --yolop_weights_path /networkhome/WMGDS/safaei_d/3_generators/models/pretrained/YOLOP/weights/End-to-end.pth \
  --beta 0.5
