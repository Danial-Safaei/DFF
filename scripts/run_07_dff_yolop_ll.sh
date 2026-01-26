#!/usr/bin/env bash
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
accelerate launch "${ROOT_DIR}/ControlNet/train_controlnet_sdxl_vkitti_dff_yolop_ll_counterfactual_20251221_183234.py" \
  --train_data_dir "${ROOT_DIR}/controlnet_vkitti_dff_yolop_ll_counterfactual_dataset" \
  --output_dir "${ROOT_DIR}/out_07_dff_yolop_ll_beta0p30_lam0p08" \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --controlnet_model_name_or_path SargeZT/sdxl-controlnet-seg \
  --image_column image --conditioning_image_column conditioning_image --caption_column prompt \
  --resolution 1248x376 \
  --train_batch_size 1 --gradient_accumulation_steps 2 \
  --gradient_checkpointing \
  --max_train_steps 60000 \
  --learning_rate 2e-05 --lr_scheduler constant_with_warmup --lr_warmup_steps 250 \
  --mixed_precision bf16 --report_to tensorboard \
  --set_grads_to_none --dataloader_num_workers 2 --allow_tf32 \
  --enable_xformers_memory_efficient_attention \
  --checkpointing_steps 999999999 \
  --yolop_ll_model_path "${ROOT_DIR}/models/pretrained/YOLOP/weights/End-to-end.pth" \
  --beta 0.3 --lambda_dff 0.08 --cf_steps 6 --cf_lr 0.02 --cf_tv 0.001 --cf_every_n 4 --cf_downsample 32
