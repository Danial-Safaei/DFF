#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
export DEBUG_GRAD_CHECK=1

source /networkhome/WMGDS/safaei_d/miniconda3/etc/profile.d/conda.sh
conda activate img2img

accelerate launch /networkhome/WMGDS/safaei_d/3_generators/training/vkitti_seg_final7/ControlNet/train_controlnet_sdxl_vkitti_dff_yolop_ll_counterfactual_20251221_183234.py \
  --train_data_dir /networkhome/WMGDS/safaei_d/3_generators/training/vkitti_seg_final7/controlnet_vkitti_dff_yolop_ll_counterfactual_dataset \
  --output_dir /networkhome/WMGDS/safaei_d/3_generators/training/vkitti_seg_final7/out_07_dff_yolop_ll_DEBUG \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --controlnet_model_name_or_path SargeZT/sdxl-controlnet-seg \
  --image_column image --conditioning_image_column conditioning_image --caption_column prompt \
  --resolution 1248x376 \
  --train_batch_size 1 --gradient_accumulation_steps 2 \
  --gradient_checkpointing \
  --max_train_steps 30 \
  --max_train_samples 64 \
  --learning_rate 2e-05 --lr_scheduler constant_with_warmup --lr_warmup_steps 0 \
  --mixed_precision bf16 --report_to tensorboard \
  --set_grads_to_none --dataloader_num_workers 2 --allow_tf32 \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps 999999999 \
  --yolop_ll_model_path /networkhome/WMGDS/safaei_d/3_generators/models/pretrained/YOLOP/weights/End-to-end.pth \
  --beta 0.3 --lambda_dff 0.08 --cf_steps 4 --cf_lr 0.02 --cf_tv 0.001 --cf_every_n 1 --cf_downsample 16
