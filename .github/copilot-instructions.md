# Copilot instructions (vkitti_seg_final7)

## Big picture
- This folder is a set of *wrappers* around HuggingFace `diffusers` SDXL ControlNet training.
- Each wrapper subclasses `VKITTITrainingBase` (imported from a sibling module `vkitti_seg/`) and:
  - reads/validates dataset + optional external model paths,
  - patches the upstream training script by **string markers** (fail-fast if markers change),
  - writes a generated training script under `ControlNet/` (timestamped name),
  - launches training and writes outputs into `out_*/`.

## Repo layout & data flow
- Training entrypoints (run these):
  - `train_k2vk_controlnet_vkitti_seg_baseline_FIXED.py`
  - `train_k2vk_controlnet_vkitti_seg_ovf_steering_CORRECTED.py`
  - `train_k2vk_controlnet_vkitti_seg_dff_steering_COUNTERFACTUAL.py` (and other `*_COUNTERFACTUAL.py` variants)
- Upstream template (do not hand-edit for experiments): `ControlNet/train_controlnet_sdxl_original.py`
- Dataset manifests (JSONL): `vkitti_*_pairs.jsonl`
- Prepared datasets for `datasets.load_dataset("json")`: `controlnet_*_dataset/metadata.jsonl`

**Dataset schema (per JSONL row):**
- `image`: KITTI input RGB image path
- `conditioning_image`: V-KITTI RGB class-segmentation map path (used as ControlNet conditioning)
- `prompt`: text prompt (often a short style prompt like `"virtual kitti style"`)

Note: some generated scripts add/use extra columns like `input_image` / `target_image` and rely on passing
`--image_column ...` / `--caption_column ...` consistently. If you change dataset columns, update both the
wrapper and the injected preprocessing block.

## Running
- Wrapper scripts are designed to be executed directly via Python, e.g.:
  - `python train_k2vk_controlnet_vkitti_seg_baseline_FIXED.py --vkitti_seg_path ... --kitti_images_path ... --output_dir out_01_baseline --steps 45000`
  - `python train_k2vk_controlnet_vkitti_seg_ovf_steering_CORRECTED.py --vkitti_seg_path ... --kitti_images_path ... --output_dir out_02_ovf_steering_beta0p70 --beta 0.7 --steps 50000`
- The generated SDXL scripts support non-square resolutions via `--resolution WxH` (e.g. `1024x512`), which
  are parsed into `args.resolution_width` / `args.resolution_height`.
- Some variants require external model paths and will **abort** if missing (e.g. OVF with `beta>0`).

## Critical conventions when editing/adding variants
- **Marker-based patching must fail fast.**
  - Variants replace/insert code using exact string markers (e.g. `import diffusers`, the dataset-prep line, the baseline MSE loss line).
  - If you update the upstream training script, you must update the wrapper markers so we don’t silently drop OVF/DFF logic.
- **Injected blocks must match context indentation.**
  - Top-level injected import blocks for the generated `ControlNet/` script must start at column 0.
  - Mid-function insertions should preserve indentation (some wrappers compute indentation from the marker line).
- **OVF/DFF losses must be differentiable w.r.t. the generator.**
  - Prefer the local `_predict_x0_from_model_pred(...)` identity instead of relying on scheduler `.step()` (which may run under `no_grad`).
- **Mixed precision & external heads:**
  - External steering models are loaded frozen (`requires_grad=False`) and run with autocast disabled to avoid dtype mismatch.

## Local module dependency
- Wrappers expect helper modules (e.g. `vkitti_training_base`, `model_loading`) to be importable from a sibling `vkitti_seg/` directory.
- Generated scripts also inject an absolute HPC path fallback like `/networkhome/WMGDS/safaei_d/3_generators/training/vkitti_seg`.

## Debugging knobs
- `DEBUG_GRAD_CHECK=1` prints a gradient norm for the OVF steering loss path (helps confirm the loss affects the generator).
- If `wandb` is installed, training logs will show up under `wandb/`.
