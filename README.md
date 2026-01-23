# DFF ControlNet Training (VKITTI/KITTI)

This repo contains the training wrappers and dataset manifests used in the paper. The core idea is to train SDXL ControlNet variants with different objectives (Baseline, OVF, DFF) and evaluate fidelity between KITTI (real) and VirtualKITTI2 (synthetic) paired images.

## What’s in this repo (minimal)

### Training scripts
These are the entrypoints you run directly with Python:

- `scripts/train_k2vk_controlnet_vkitti_seg_baseline_FIXED.py` — Baseline reconstruction-only ControlNet training.
- `scripts/train_k2vk_controlnet_vkitti_seg_ovf_steering_CORRECTED.py` — OVF training (steering head).
- `scripts/train_k2vk_controlnet_vkitti_seg_ovf_yolop_da_CORRECTED.py` — OVF training (YOLOP drivable area head).
- `scripts/train_k2vk_controlnet_vkitti_seg_ovf_yolop_ll_CORRECTED.py` — OVF training (YOLOP lane line head).
- `scripts/train_k2vk_controlnet_vkitti_seg_dff_steering_COUNTERFACTUAL.py` — DFF training with CF‑XAI (steering).
- `scripts/train_k2vk_controlnet_vkitti_seg_dff_yolop_da_COUNTERFACTUAL.py` — DFF training with CF‑XAI (drivable area).
- `scripts/train_k2vk_controlnet_vkitti_seg_dff_yolop_ll_COUNTERFACTUAL.py` — DFF training with CF‑XAI (lane lines).

### Dataset manifests (JSONL)
Each JSONL row contains:
- `image`: KITTI RGB image path
- `conditioning_image`: VKITTI RGB class-segmentation map path
- `prompt`: short text prompt (e.g., “virtual kitti style”)

Manifests used by each variant:
- `data/vkitti_baseline_pairs.jsonl`
- `data/vkitti_ovf_steering_corrected_pairs.jsonl`
- `data/vkitti_ovf_yolop_da_corrected_pairs.jsonl`
- `data/vkitti_ovf_yolop_ll_corrected_pairs.jsonl`
- `data/vkitti_dff_steering_counterfactual_pairs.jsonl`
- `data/vkitti_dff_yolop_da_counterfactual_pairs.jsonl`
- `data/vkitti_dff_yolop_ll_counterfactual_pairs.jsonl`

## How the scripts work
All training wrappers follow the same pattern:
1. Validate dataset paths and optional external model paths.
2. Patch the upstream SDXL ControlNet training script using exact string markers.
3. Generate a timestamped script under `ControlNet/`.
4. Launch training and write outputs under `out_*` directories.

Marker‑based patching fails fast if the upstream script changes to prevent silent divergence.

## Running
Example usage (adjust paths):

```bash
python scripts/train_k2vk_controlnet_vkitti_seg_baseline_FIXED.py \
  --vkitti_seg_path /path/to/vkitti_seg \
  --kitti_images_path /path/to/kitti \
  --output_dir out_01_baseline \
  --steps 45000
```

```bash
python scripts/train_k2vk_controlnet_vkitti_seg_ovf_steering_CORRECTED.py \
  --vkitti_seg_path /path/to/vkitti_seg \
  --kitti_images_path /path/to/kitti \
  --output_dir out_02_ovf_steering_beta0p70 \
  --beta 0.7 \
  --steps 50000
```

## Fidelity thresholds (paper)
Fidelity thresholds follow domain standards and calibration distributions: 
$\varepsilon_{\mathrm{out}}$ uses a 0.7 similarity cutoff (e.g., $\mathrm{IoU}\!\ge\!0.7$).

Per‑SUT DFF thresholds (95th percentile on calibration split):
- Steering: $\varepsilon_{95}=0.08738442$
- YOLOP‑DA: $\varepsilon_{95}=0.6990833$
- YOLOP‑LL: $\varepsilon_{95}=0.7362408$

## Notes
- DFF is computed using CF‑XAI pooled decisive maps (16×16) with $K=80$ counterfactual seeds.
- OVF for segmentation uses IoU for the paper’s main results.

## Contact
For questions or replication issues, open an issue or contact the author.
