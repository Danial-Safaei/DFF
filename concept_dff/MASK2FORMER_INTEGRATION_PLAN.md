# DFF + Mask2Former Concept-Importance Integration Plan

## Scope
Create three new training wrappers that combine the existing DFF objectives with Mask2Former-derived concept importance. Mask2Former is treated as the authoritative concept source (non-differentiable) and is never used inside the gradient path. All new wrappers keep fail-fast marker patching and follow the existing wrapper architecture.

## ControlNet behavior (denoising context)
ControlNet is a conditioned denoiser: it runs the same diffusion denoising process as SDXL but injects conditioning features into the U-Net at each step. Conditioning therefore *biases the denoising trajectory* toward the structure in the control signal. High-fidelity Mask2Former maps will strongly steer semantic layout early in denoising; stronger conditioning improves class-consistent geometry but can reduce texture/style diversity.

## New artifacts
- Plan (this file): MASK2FORMER_INTEGRATION_PLAN.md
- New wrappers:
  - scripts/train_k2vk_controlnet_vkitti_seg_dff_steering_CONCEPT.py
  - scripts/train_k2vk_controlnet_vkitti_seg_dff_yolop_da_CONCEPT.py
  - scripts/train_k2vk_controlnet_vkitti_seg_dff_yolop_ll_CONCEPT.py

## Integration philosophy (most faithful)
The most faithful integration is to compute concept importance from Mask2Former outputs directly and use that to weight DFF losses. This preserves the exact semantics of Mask2Former while keeping training differentiable (Mask2Former is run under no-grad; its outputs gate losses but do not require gradients).

## Concept-importance definition (training-time)
1. Compute counterfactual change maps (diff_real, diff_pred) from the existing DFF inner loops.
2. Run Mask2Former on real and predicted decoded images to obtain semantic segmentation IDs.
3. Aggregate concept importance per class by averaging the diff map inside each class region.
4. Apply two penalties:
   - Concept alignment loss: MSE between concept-importance vectors (pred vs real).
   - Undesired concept penalty: mean importance on a configurable list of “undesired” classes.

## Loss composition (per step)
Total loss per step:
- L_total = L_recon + β·L_OVF + λ_dff·L_DFF + λ_ci·(L_ci_align + α_undesired·L_ci_undesired)

Where:
- L_ci_align compares concept-importance vectors.
- L_ci_undesired penalizes saliency mass on undesired classes.

## CLI controls (all three wrappers)
- --mask2former_model_id (default: facebook/mask2former-swin-large-cityscapes-semantic)
- --lambda_concept (λ_ci)
- --undesired_concepts (comma-separated Cityscapes names)
- --undesired_scale (α_undesired)
- --concept_every_n (compute concept losses every N steps)
- --concept_downsample (downsample size for diff-map aggregation)

## Validation and logging
- Log per-step concept losses and top-k concept weights (real vs pred) in accelerator logs.
- Add DEBUG_GRAD_CHECK hooks to confirm concept losses do not break gradient flow.
- Fail fast if Mask2Former or transformers are missing.

## Next refinement checkpoints
- Finalize undesired concept list (default: sky, vegetation, building, terrain).
- Decide whether to compute Mask2Former at full resolution or training resolution.
- Confirm whether to cache segmentations for the real targets.
