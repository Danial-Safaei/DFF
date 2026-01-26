# Living Report — vkitti_seg_final7

Last updated: 2026-01-08

## Purpose
A single continuously-updated, auditable report for this workspace.

This file is intended to be appended/updated after every meaningful change or investigation (training runs, notebook edits, metric definition changes, bugfixes, new outputs).

## How to update (discipline)
When you (or a future agent) change anything substantial, add an entry to **Update Log** with:
- date/time
- what changed
- why
- where (paths)
- impact on reproducibility/results
- any new TODOs/risks

## Workspace snapshot
- Notebook: `dff_paper_experiments_max5cells.ipynb`
- Training wrappers (entrypoints):
  - `train_k2vk_controlnet_vkitti_seg_baseline_FIXED.py`
  - `train_k2vk_controlnet_vkitti_seg_ovf_steering_CORRECTED.py`
  - `train_k2vk_controlnet_vkitti_seg_ovf_yolop_da_CORRECTED.py`
  - `train_k2vk_controlnet_vkitti_seg_ovf_yolop_ll_CORRECTED.py`
  - `train_k2vk_controlnet_vkitti_seg_dff_steering_COUNTERFACTUAL.py`
  - `train_k2vk_controlnet_vkitti_seg_dff_yolop_da_COUNTERFACTUAL.py`
  - `train_k2vk_controlnet_vkitti_seg_dff_yolop_ll_COUNTERFACTUAL.py`
- Upstream training template: `ControlNet/train_controlnet_sdxl_original.py`
- Run scripts: `run_01_*.sh` … `run_07_*.sh`
- Dataset manifests: `controlnet_*_dataset/metadata.jsonl`
- Outputs: `out_*/` (ControlNetModel checkpoints)

## Data schema (manifest JSONL)
Each row is expected to include:
- `image`: path to KITTI RGB
- `conditioning_image`: path to VKITTI RGB class-segmentation map (conditioning)
- `prompt`: short prompt string

## Key audit findings (current)
### 1) Notebook runtime blockers (must-fix before running)
Notebook: `dff_paper_experiments_max5cells.ipynb`

Observed undefined/missing symbols/imports in later cells (will crash at runtime):
- `K_CF_PAPER` (config defines `K_CF`, not `K_CF_PAPER`)
- `_to_torch_image`, `_to_yolop_input`
- `tF` (used like `torch.nn.functional` but not imported/aliased)
- `pd` / `plt` / `sns` / `spearmanr` / `sys` referenced without imports

Impact: Notebook is not currently executable end-to-end without edits.

### 2) Variant key mismatch in Figure 3 annotations
In the Figure 3 grid, variant keys are derived via `key.lower().replace("-", "_")`.
- Example: `"OVF-Steer" -> "ovf_steer"` but the actual variant key is `"ovf_steering"`.

Impact: metrics overlays may silently fail/mis-annotate even if plotting succeeds.

### 3) Pilot frame selection robustness
Table III selects pilot rows by `PILOT_FRAME_ID` substring match on `image_path`.
- If frame IDs repeat across scenes, this can pick the wrong row.
- Split selection earlier properly checks `PILOT_SCENE` + `PILOT_FRAME_ID`.

Impact: Table III may report wrong pilot-frame metrics.

### 4) Conceptual mismatch: training-time DFF vs notebook-time DFF
Training scripts implement a differentiable DFF-style objective based on counterfactual/change-map consistency (periodic, differentiable unrolling on the predicted branch).
Notebook DFF is a CF-XAI “minimal deletion” decisive-map distance (mask optimization against blur baseline).

Impact: “DFF” is overloaded; training objective and evaluation metric are not the same definition.

### 5) Training pipeline architecture (repro constraints)
Wrappers patch the upstream diffusers training template using brittle string markers and *fail fast* if markers change.
- Good: avoids silently dropping OVF/DFF logic.
- Constraint: updating `ControlNet/train_controlnet_sdxl_original.py` requires updating wrapper markers.

## Repro checklist (current)
- [ ] Notebook executes end-to-end without manual fixes
- [ ] All required packages installed in notebook kernel (`torch`, `diffusers`, `transformers`, `accelerate`, `safetensors`, plus metrics/plotting deps)
- [ ] SUT paths resolvable (steering model; YOLOP optional)
- [ ] Out dirs contain `config.json` + `diffusion_pytorch_model.safetensors`
- [ ] Pilot frame logic selects correct scene+frame

## TODO (next concrete fixes)
1. Patch notebook to define/import the missing symbols and correct naming (`K_CF_PAPER` → `K_CF`).
2. Fix Figure 3 variant mapping so overlays match variant keys.
3. Tighten Table III pilot filtering to constrain by both `PILOT_SCENE` and `PILOT_FRAME_ID`.

## Update log
### 2026-01-08
- Created this living report file.
- Seeded with the current deep-audit findings of the notebook + training pipeline.
