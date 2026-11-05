# DFF ControlNet Training (VKITTI/KITTI)

This repo contains the training wrappers and dataset manifests used in the paper. The core idea is to train SDXL ControlNet variants with different objectives (Baseline, OVF, DFF) and evaluate fidelity between KITTI (real) and VirtualKITTI2 (synthetic) paired images.

## Paper

**Quantifying Fidelity: A Decisive Feature Approach to Comparing Synthetic and Real Imagery**
Danial Safaei, Siddartha Khastgir, Mohsen Alirezaei, Jeroen Ploeg, Chih-Hong Cheng, Son Tong, Xingyu Zhao

Published in the *2026 IEEE Intelligent Vehicles Symposium (IV)*, pp. 847-854, Plymouth, MI, USA, June 2026.

- IEEE Xplore (DOI): https://doi.org/10.1109/IV66570.2026.11624013
- arXiv preprint: https://arxiv.org/abs/2512.16468

### Cite this work
If you use this repository, please cite the published version:

```bibtex
@inproceedings{safaei2026quantifying,
  title     = {Quantifying Fidelity: A Decisive Feature Approach to Comparing Synthetic and Real Imagery},
  author    = {Safaei, Danial and Khastgir, Siddartha and Alirezaei, Mohsen and Ploeg, Jeroen and Cheng, Chih-Hong and Tong, Son and Zhao, Xingyu},
  booktitle = {2026 IEEE Intelligent Vehicles Symposium (IV)},
  pages     = {847--854},
  year      = {2026},
  address   = {Plymouth, MI, USA},
  publisher = {IEEE},
  doi       = {10.1109/IV66570.2026.11624013}
}
```

### Follow-up work
**C-DFF: Concept-Level Diagnostics for Mechanism Parity in Simulation-Based Validation of Autonomous Driving Perception**
Danial Safaei, Siddartha Khastgir, Matthew Higgins, Jeroen Ploeg, Mohsen Alirezaei, Son Tong, Ludo Gielen
*2026 IEEE 29th International Conference on Intelligent Transportation Systems (ITSC)*, Naples, Italy (in press).

- arXiv: https://arxiv.org/abs/2610.33760

C-DFF expresses decisive evidence, extracted with its own perturbation-based estimator, in terms of semantic scene concepts such as road, building and sky, to show which concepts a model relies on more, or less, in simulation. Its code is not part of this repository.

## What’s in this repo (minimal)

### Repository layout
- `scripts/` — runnable training wrappers and helper scripts.
- `data/` — dataset manifest JSONL files.
- `vkitti_seg/` — core helpers (training base + loaders).
- `models/` — external SUT assets (steering arch/weights, YOLOP weights).
- `third_party/YOLOP/` — YOLOP repo (download separately; see below).
- `ControlNet/` — upstream SDXL template + generated scripts.
- `out_*/` — training outputs (not tracked).

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
### External model setup (required)
1) Steering architecture is vendored under `models/architectures/`.
2) Download YOLOP repo into `third_party/YOLOP`:
```bash
git clone https://github.com/hustvl/YOLOP third_party/YOLOP
```
3) Obtain the system-under-test weights (see "Data availability" below) and place them at:
    - `models/weights/final_model.pth`
    - `models/pretrained/YOLOP/weights/End-to-end.pth`

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

## Data availability
The repository includes only JSONL manifests under `data/`. The underlying KITTI and VirtualKITTI2 images are not distributed here. Obtain them from their official sources and pass local paths via the CLI flags shown above.

## Baseline dataset helper
`create_kitti_to_vkitti_dataset.py` creates the baseline KITTI→V-KITTI dataset structure expected by the baseline wrapper.

## Reproducibility checklist
- Use the same dataset split and pairing manifests from `data/`.
- Use the same SUT (steering/YOLOP-DA/YOLOP-LL) and threshold calibration described in the paper.
- Ensure the upstream SDXL ControlNet template in `ControlNet/` matches the expected patch markers.
- Record the random seed, $K$, and N_SAMPLES used for each evaluation run.

## License
This repository is released under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

## Contact
For questions or replication issues, open an issue or contact the author.
