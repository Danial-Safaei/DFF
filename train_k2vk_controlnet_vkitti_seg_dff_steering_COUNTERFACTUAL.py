#!/usr/bin/env python3
"""
V-KITTI SEGMENTATION + DFF STEERING

Differentiable Feature Fidelity (DFF) + OVF for Steering.
We replace non-differentiable counterfactual loops with differentiable saliency-based maps
so that the DFF term backpropagates into the generator.

Objective: L_total = L_reconstruction + β * L_OVF + λ * L_DFF_SALIENCY

Usage:
    python train_k2vk_controlnet_vkitti_seg_dff_steering_COUNTERFACTUAL.py \
        --vkitti_seg_path /path/to/vkitti/segmentation \
        --kitti_images_path /path/to/kitti/images \
        --output_dir /path/to/output \
        --steering_model_path /path/to/steering_model.pth \
    --beta 0.3 --lambda_dff 0.08 \
        --steps 60000
"""

import os
import sys
import pathlib

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_VKITTI_SEG_DIR = _THIS_DIR.parent / "vkitti_seg"
if str(_VKITTI_SEG_DIR) not in sys.path:
    sys.path.insert(0, str(_VKITTI_SEG_DIR))

from vkitti_training_base import VKITTITrainingBase, create_robust_argument_parser


class VKITTIDFFCounterfactualSteeringTrainer(VKITTITrainingBase):
    """V-KITTI + DFF (saliency) Steering trainer with OVF + DFF losses."""

    def __init__(self):
        super().__init__("dff_steering_counterfactual")

    def validate_model_paths(self, **kwargs) -> bool:
        """Validate Steering Counterfactual-specific model paths."""
        steering_path = kwargs.get('steering_model_path')
        if not steering_path:
            self.logger.error("❌ Steering model path required for DFF training")
            return False

        if not os.path.exists(steering_path):
            self.logger.error(f"❌ Steering model not found: {steering_path}")
            return False

        self.logger.info("✅ Counterfactual X-AI technique will be used (no external deps required)")
        self.logger.info(f"✅ Steering model validated: {steering_path}")
        return True

    def apply_variant_modifications(self, base_content: str, **kwargs) -> str:
        """
        Apply DFF Steering (saliency) modifications using robust template approach.
        """
        self.logger.info("🔧 Applying V-KITTI + DFF Steering (saliency) modifications...")

        # CRITICAL: Marker-based patching must fail fast.
        def must_contain(content: str, marker: str, label: str) -> None:
            if marker not in content:
                raise RuntimeError(f"Required patch marker not found ({label}): {marker!r}")

        beta = kwargs.get('beta', 0.3)
        lambda_dff = kwargs.get('lambda_dff', 0.08)
        modifications = []

        # 1) Add imports and counterfactual helpers
        import_marker = "import diffusers"
        must_contain(base_content, import_marker, "diffusers import")
        dff_imports = '''import diffusers

# V-KITTI + DFF Steering Counterfactual Imports
# NOTE: This block is injected at top-level into the generated HF training script.
# It MUST NOT start with indentation, otherwise the generated script will be invalid Python.
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import os
import sys
from pathlib import Path
import torchvision.transforms as transforms

# Make sure the generated script (written under ControlNet/) can import our local helpers.
# Rely on an absolute path first (HPC layout), with a conservative relative fallback.
_ABS_VKITTI_SEG = "/networkhome/WMGDS/safaei_d/3_generators/training/vkitti_seg"
if _ABS_VKITTI_SEG not in sys.path:
    sys.path.insert(0, _ABS_VKITTI_SEG)
try:
    _here = Path(__file__).resolve()
    _fallback = _here.parent.parent.parent / "training" / "vkitti_seg"
    if str(_fallback) not in sys.path:
        sys.path.append(str(_fallback))
except Exception:
    pass
from model_loading import load_steering_model

# V-KITTI Class Mapping
VKITTI_CLASS_MAPPING = {
    'Terrain': (210, 0, 200, 0), 'Sky': (90, 200, 255, 1), 'Tree': (0, 199, 0, 2),
    'Vegetation': (90, 240, 0, 3), 'Building': (140, 140, 140, 4), 'Road': (100, 60, 100, 5),
    'GuardRail': (250, 100, 255, 6), 'TrafficSign': (255, 255, 0, 7), 'TrafficLight': (200, 200, 0, 8),
    'Pole': (255, 130, 0, 9), 'Misc': (80, 80, 80, 10), 'Truck': (160, 60, 60, 11),
    'Car': (255, 127, 80, 12), 'Van': (0, 139, 139, 13), 'Undefined': (0, 0, 0, 14)
}

RGB_TO_INDEX = {(r, g, b): idx for _, (r, g, b, idx) in VKITTI_CLASS_MAPPING.items()}

def rgb_to_class_indices(rgb_seg_map):
    """Convert V-KITTI RGB segmentation to class indices."""
    if hasattr(rgb_seg_map, 'convert'):
        rgb_seg_map = np.array(rgb_seg_map)

    height, width = rgb_seg_map.shape[:2]
    class_indices = np.zeros((height, width), dtype=np.uint8)

    for (r, g, b), class_idx in RGB_TO_INDEX.items():
        mask = (
            (np.abs(rgb_seg_map[:, :, 0] - r) <= 2) &
            (np.abs(rgb_seg_map[:, :, 1] - g) <= 2) &
            (np.abs(rgb_seg_map[:, :, 2] - b) <= 2)
        )
        class_indices[mask] = class_idx

    return class_indices

# ---- Robust tensor/scheduler/decoder helpers (aligned with CORRECTED OVF scripts) ----
def _extract_first_tensor(obj):
    import torch as _torch
    if isinstance(obj, _torch.Tensor):
        return obj
    if hasattr(obj, "sample"):
        return _extract_first_tensor(getattr(obj, "sample"))
    if isinstance(obj, dict):
        for v in obj.values():
            t = _extract_first_tensor(v)
            if t is not None:
                return t
        return None
    if isinstance(obj, (list, tuple)):
        for v in obj:
            t = _extract_first_tensor(v)
            if t is not None:
                return t
        return None
    return None

def _safe_scheduler_pred_x0(noise_scheduler, model_pred, timesteps, noisy_latents):
    out = noise_scheduler.step(model_pred, timesteps, noisy_latents)
    candidates = []
    if hasattr(out, "pred_original_sample"):
        candidates.append(out.pred_original_sample)
    if hasattr(out, "prev_sample"):
        candidates.append(out.prev_sample)
    candidates.append(out)
    t = None
    for c in candidates:
        t = _extract_first_tensor(c)
        if t is not None:
            break
    if t is None:
        raise RuntimeError("Scheduler step produced no tensor-like output")
    return t

def _predict_x0_from_model_pred(noise_scheduler, model_pred, timesteps, noisy_latents):
    """Differentiable x0 reconstruction.

    DFF/OVF must be able to backpropagate into the generator. Many schedulers implement
    `.step()` under `torch.no_grad()` (inference). If that happens, DFF/OVF becomes a no-op.
    """
    pred_type = getattr(getattr(noise_scheduler, "config", None), "prediction_type", "epsilon")
    try:
        alphas = noise_scheduler.alphas_cumprod.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
        a_t = alphas[timesteps]
        shape = (-1,) + (1,) * (noisy_latents.dim() - 1)
        a_t = a_t.reshape(shape)
        sqrt_a = a_t.sqrt()
        sqrt_one_minus = (1.0 - a_t).sqrt()
        if pred_type in ("epsilon", "eps"):
            return (noisy_latents - sqrt_one_minus * model_pred) / sqrt_a
        if pred_type in ("v_prediction", "v"):
            return sqrt_a * noisy_latents - sqrt_one_minus * model_pred
    except Exception:
        pass
    return _safe_scheduler_pred_x0(noise_scheduler, model_pred, timesteps, noisy_latents)

def _safe_vae_decode(vae, latents):
    # Decode in fp32 to avoid bf16 input vs fp32 bias errors inside VAE convs.
    latents_f32 = latents.float()
    decoded = vae.decode(latents_f32 / vae.config.scaling_factor)
    t = _extract_first_tensor(decoded)
    if t is None:
        decoded2 = vae.decode(latents_f32 / vae.config.scaling_factor, return_dict=False)
        t = _extract_first_tensor(decoded2)
    if t is None:
        raise RuntimeError("VAE decode produced no tensor output")
    return t

# Counterfactual helpers (DFF)
def _tv_loss(x, weight=1.0):
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return weight * (dx.abs().mean() + dy.abs().mean())

def _clamp01(x):
    return x.clamp(0.0, 1.0)

def steering_counterfactual(model, x_in, steps=80, lr=3e-2, tv=1e-3, enable_graph: bool = False):
    """Generate a counterfactual that flips steering, with truncated higher-order graph.
    - Early inner steps: no graph retention (low memory)
    - Final step: create_graph=True when enable_graph to allow backprop to predicted path
    """
    # Use differentiable origin only when requested
    x0 = x_in if enable_graph else x_in.detach()
    # Reference target (no graph needed)
    with torch.no_grad():
        y_orig = model(x0.float()).detach()
        target = -torch.sign(y_orig) * (y_orig.abs() + 0.5)

    x_cf = x0.clone()
    steps_eff = int(steps)
    if enable_graph:
        # Capped unrolling to keep memory bounded while preserving a real gradient path.
        steps_eff = min(steps_eff, 8)
    for _ in range(max(1, steps_eff)):
        if enable_graph:
            x_cf = x_cf.clone()
            x_cf.requires_grad_(True)
        else:
            x_cf = x_cf.detach().requires_grad_(True)
        x_cf32 = x_cf.float()
        y_pred = model(x_cf32)
        loss_inner = F.mse_loss(y_pred, target) + _tv_loss(x_cf32, tv)
        grad = torch.autograd.grad(loss_inner, x_cf, retain_graph=enable_graph, create_graph=enable_graph)[0]
        x_cf = (x_cf - lr * grad).clamp(0.0, 1.0)

    diff = (x_cf - x0).abs().mean(dim=1, keepdim=True)
    return (x_cf, diff) if enable_graph else (x_cf.detach(), diff.detach())

# Simple Steering model class for compatibility
class SimpleSteeringModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_layer = torch.nn.Linear(1, 1)

    def forward(self, x):
        b = x.shape[0]
        x_flat = x.view(b, -1).mean(dim=1, keepdim=True)
        return self.dummy_layer(x_flat)

# Global variables for DFF Steering
STEERING_MODEL = None
OVF_BETA = 0.3
LAMBDA_DFF = 0.08
CF_STEER_STEPS = 80
CF_STEER_LR = 3e-2
CF_STEER_TV = 1e-3
CF_EVERY_N = 1
CF_DOWNSAMPLE = 16
        '''

        modifications.append((import_marker, dff_imports))

        # 2) Setup models/args
        args_marker = "if args.seed is not None:"
        must_contain(base_content, args_marker, "seed block")
        dff_setup = '''
    # DFF Steering Model Setup
    global STEERING_MODEL, OVF_BETA, LAMBDA_DFF, CF_STEER_STEPS, CF_STEER_LR, CF_STEER_TV, CF_EVERY_N, CF_DOWNSAMPLE

    steering_model_path = getattr(args, 'steering_model_path', None)
    if not steering_model_path or not os.path.exists(steering_model_path):
        raise RuntimeError(f"DFF steering requires a valid --steering_model_path; got: {steering_model_path}")

    # Resolve the steering architecture robustly. Prefer project absolute path.
    proj_arch = Path("/networkhome/WMGDS/safaei_d/3_generators/models/architectures/For_steering_angle_prediction_tweaked_v3.py")
    if proj_arch.exists():
        arch_path = str(proj_arch)
    else:
        arch_path = str(Path(__file__).parent.parent / "models" / "architectures" / "For_steering_angle_prediction_tweaked_v3.py")

    print("🔄 Loading steering model via load_steering_model; arch=", arch_path, "; weights=", steering_model_path)
    # Match dtype to accelerator mixed precision to avoid (bf16/fp16) input vs fp32 bias errors
    try:
        import torch as _torch
        mp = getattr(accelerator, "mixed_precision", "no")
        _ext_dtype = _torch.bfloat16 if mp == "bf16" else (_torch.float16 if mp == "fp16" else _torch.float32)
    except Exception:
        _ext_dtype = None
    STEERING_MODEL = load_steering_model(arch_path, steering_model_path, accelerator.device, dtype=_ext_dtype)
    if STEERING_MODEL is None or getattr(STEERING_MODEL, "is_dummy", False):
        raise RuntimeError("Failed to load a real steering model; refusing to run DFF/OVF with a dummy/fallback model.")
    print("✅ Steering model ready")

    OVF_BETA = getattr(args, 'beta', 0.3)
    LAMBDA_DFF = getattr(args, 'lambda_dff', 0.08)
    try:
        CF_STEER_STEPS = int(getattr(args, 'cf_steps', CF_STEER_STEPS))
        CF_STEER_LR = float(getattr(args, 'cf_lr', CF_STEER_LR))
        CF_STEER_TV = float(getattr(args, 'cf_tv', CF_STEER_TV))
        CF_EVERY_N = int(getattr(args, 'cf_every_n', CF_EVERY_N))
        CF_DOWNSAMPLE = int(getattr(args, 'cf_downsample', CF_DOWNSAMPLE))
    except Exception:
        pass
    print("🎯 OVF Beta (steering output consistency):", OVF_BETA)
    print("🎯 Lambda DFF (consistency weight):", LAMBDA_DFF)
    print("🎯 DFF (counterfactual) settings: steps=", CF_STEER_STEPS, ", lr=", CF_STEER_LR, ", tv=", CF_STEER_TV, ", every_n=", CF_EVERY_N, ", down=", CF_DOWNSAMPLE)

    if args.seed is not None:'''

        modifications.append((args_marker, dff_setup))

        # 3) Preprocessing remains the same (pure segmentation conditioning)
        dataset_prep_marker = "train_dataset = prepare_train_dataset(train_dataset, accelerator)"
        must_contain(base_content, dataset_prep_marker, "dataset prep")
        dff_preprocessing = '''
    # V-KITTI + DFF (name only) Steering Counterfactual Segmentation Preprocessing
    from datasets import Image as DatasetImage
    train_dataset = train_dataset.cast_column(args.image_column, DatasetImage())
    train_dataset = train_dataset.cast_column(args.conditioning_image_column, DatasetImage())

    def preprocess_vkitti_segmentation_dff_counterfactual(examples):
        conditioning_images = []
        for seg_path in examples[args.conditioning_image_column]:
            if isinstance(seg_path, str):
                seg_img = Image.open(seg_path).convert('RGB')
            else:
                seg_img = seg_path.convert('RGB')

            seg_resized = seg_img.resize((args.resolution_width, args.resolution_height), Image.NEAREST)
            conditioning_images.append(seg_resized)

        examples[args.conditioning_image_column] = conditioning_images
        return examples

    train_dataset = train_dataset.map(preprocess_vkitti_segmentation_dff_counterfactual, batched=True)
    print("✅ Applied V-KITTI+DFF Steering (counterfactual) segmentation preprocessing to", len(train_dataset), "samples")

    train_dataset = prepare_train_dataset(train_dataset, accelerator)'''

        modifications.append((dataset_prep_marker, dff_preprocessing))

        # 4) Loss modification: Reconstruction + OVF steering + DFF (name only) counterfactual changemap
        loss_marker = 'loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")'
        dff_loss = '''
                # V-KITTI + DFF (name only) Steering Counterfactual Enhanced Loss
                reconstruction_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss = reconstruction_loss

                if STEERING_MODEL is not None and (OVF_BETA > 0 or LAMBDA_DFF > 0):
                    try:
                        # Real and predicted images using safe helpers (differentiable CF for predicted path)
                        x_tgt_normalized = batch["pixel_values"]
                        # Differentiable x0 reconstruction is critical for DFF/OVF to affect the generator.
                        pred_original_sample = _predict_x0_from_model_pred(noise_scheduler, model_pred, timesteps, noisy_latents)
                        x_tgt_pred_normalized = _safe_vae_decode(vae, pred_original_sample)

                        # Normalize to [0,1] for steering model
                        x_tgt = (x_tgt_normalized * 0.5) + 0.5
                        x_tgt_pred = (x_tgt_pred_normalized * 0.5) + 0.5

                        # Resize for steering model input (66x200)
                        x_tgt_steer = F.interpolate(x_tgt, size=(66, 200), mode='bilinear', align_corners=False)
                        x_tgt_pred_steer = F.interpolate(x_tgt_pred, size=(66, 200), mode='bilinear', align_corners=False)

                        # 1) OVF: steering outputs consistency (real no-grad, predicted with grad)
                        try:
                            with torch.no_grad():
                                with torch.autocast("cuda", enabled=False):
                                    real_steer = STEERING_MODEL(x_tgt_steer.detach().float())
                            with torch.autocast("cuda", enabled=False):
                                pred_steer = STEERING_MODEL(x_tgt_pred_steer.float())
                            if real_steer.dim() > 2: real_steer = real_steer.view(real_steer.shape[0], -1)
                            if pred_steer.dim() > 2: pred_steer = pred_steer.view(pred_steer.shape[0], -1)
                            loss_ovf_steer = F.mse_loss(real_steer, pred_steer)
                        except Exception as ovf_e:
                            loss_ovf_steer = torch.tensor(0.0, device=accelerator.device)
                            if accelerator.is_local_main_process:
                                print(f"⚠️ OVF steering calculation failed: {ovf_e}")

                        # 2) DFF counterfactual change-map consistency (differentiable wrt predicted path)
                        try:
                            compute_cf = True
                            try:
                                gs = int(global_step) if not isinstance(global_step, int) else global_step
                                compute_cf = (gs % max(1, CF_EVERY_N)) == 0
                            except Exception:
                                compute_cf = True

                            if compute_cf:
                                # CF-X-AI: Enable gradient backpropagation through counterfactual generation
                                # Clear cache before expensive CF computation
                                torch.cuda.empty_cache()

                                # Real CF: non-differentiable (baseline)
                                # Predicted CF: DIFFERENTIABLE - enables CF-X-AI gradient flow back to generator
                                x_cf_real, diff_real = steering_counterfactual(STEERING_MODEL, x_tgt_steer.detach(), steps=CF_STEER_STEPS, lr=CF_STEER_LR, tv=CF_STEER_TV, enable_graph=False)
                                x_cf_pred, diff_pred = steering_counterfactual(STEERING_MODEL, x_tgt_pred_steer, steps=CF_STEER_STEPS, lr=CF_STEER_LR, tv=CF_STEER_TV, enable_graph=True)

                                ds = max(4, int(CF_DOWNSAMPLE))
                                diff_real_ds = F.adaptive_avg_pool2d(diff_real, (ds, ds))
                                diff_pred_ds = F.adaptive_avg_pool2d(diff_pred, (ds, ds))
                                loss_dff_cf = F.mse_loss(diff_real_ds, diff_pred_ds)

                                # Optional proof-of-gradient (disabled by default). Enable with DEBUG_GRAD_CHECK=1.
                                # Confirms the DFF counterfactual term backprops to the generator (via model_pred).
                                if os.environ.get("DEBUG_GRAD_CHECK", "0").lower() in ("1", "true", "yes"):
                                    try:
                                        g = torch.autograd.grad(loss_dff_cf, model_pred, retain_graph=True, allow_unused=True)[0]
                                        if g is None:
                                            print("[DFF][steering] grad(model_pred) is None (DFF may not affect generator)")
                                        else:
                                            gn = float(g.detach().norm().item())
                                            if accelerator.is_local_main_process:
                                                print(f"[DFF][steering] grad_norm(model_pred)={gn:.6g}")
                                    except Exception as _ge:
                                        if accelerator.is_local_main_process:
                                            print(f"[DFF][steering] grad check failed: {_ge}")

                                # Clear cache after CF
                                torch.cuda.empty_cache()
                            else:
                                loss_dff_cf = torch.tensor(0.0, device=accelerator.device)
                        except Exception as cf_e:
                            loss_dff_cf = torch.tensor(0.0, device=accelerator.device)
                            if accelerator.is_local_main_process:
                                print(f"⚠️ DFF (counterfactual) calculation failed: {cf_e}")

                        conditioning_complexity = torch.std(batch["conditioning_pixel_values"]).item()
                        adaptive_beta = OVF_BETA * (1.0 + 0.5 * conditioning_complexity)
                        road_content = (batch["conditioning_pixel_values"] > 0.8).float().mean().item()
                        adaptive_lambda = LAMBDA_DFF * (1.0 + 0.4 * road_content)

                        # Combine losses
                        loss = reconstruction_loss + adaptive_beta * loss_ovf_steer + adaptive_lambda * loss_dff_cf

                        if accelerator.is_local_main_process:
                            logs = {
                                "reconstruction_loss": reconstruction_loss.detach().item(),
                                "ovf_steering_output_loss": loss_ovf_steer.detach().item(),
                                "dff_counterfactual_changemap_loss": loss_dff_cf.detach().item(),
                                "adaptive_beta": adaptive_beta,
                                "adaptive_lambda": adaptive_lambda,
                                "conditioning_complexity": conditioning_complexity,
                                "road_content": road_content,
                                "total_loss": loss.detach().item(),
                            }
                            accelerator.log(logs, step=global_step)

                    except Exception as e:
                        if accelerator.is_local_main_process:
                            print(f"⚠️ DFF Steering (counterfactual) loss failed, using reconstruction loss: {e}")
                        loss = reconstruction_loss'''

        modifications.append((loss_marker, dff_loss))

        # 5) Extend parser with steering args
        args_parser_marker = '    if input_args is not None:'
        dff_args_addition = '''    # DFF Steering Counterfactual Arguments
    parser.add_argument("--steering_model_path", type=str, required=False, help="Path to steering prediction model")
    parser.add_argument("--beta", type=float, default=0.3, help="OVF steering loss weight")
    parser.add_argument("--lambda_dff", type=float, default=0.08, help="DFF counterfactual changemap consistency weight")
    parser.add_argument("--cf_steps", type=int, default=80, help="DFF (name only) counterfactual inner steps")
    parser.add_argument("--cf_lr", type=float, default=3e-2, help="DFF (name only) counterfactual step size")
    parser.add_argument("--cf_tv", type=float, default=1e-3, help="DFF (name only) counterfactual TV strength")
    parser.add_argument("--cf_every_n", type=int, default=1, help="Compute DFF (name only) every N steps (>=1)")
    parser.add_argument("--cf_downsample", type=int, default=16, help="Downsample DFF (name only) maps to NxN before loss")

    if input_args is not None:'''
        modifications.append((args_parser_marker, dff_args_addition))

        # Apply all modifications on base template
        modified_content = base_content
        for marker, replacement in modifications:
            if marker in modified_content:
                modified_content = modified_content.replace(marker, replacement)
                self.logger.info(f"✅ Applied DFF Steering (saliency) modification: {marker[:30]}...")
            else:
                raise RuntimeError(f"Required patch marker not found (DFF steering): {marker[:120]}")

        self.logger.info("✅ DFF (name only) Steering Counterfactual modifications applied successfully")
        return modified_content

    def get_training_arguments(self, **kwargs) -> list:
        """Get DFF (name only) Steering Counterfactual training arguments."""
        args = []
        if 'steering_model_path' in kwargs:
            args.extend(["--steering_model_path", kwargs['steering_model_path']])
        if 'beta' in kwargs:
            args.extend(["--beta", str(kwargs['beta'])])
        if 'lambda_dff' in kwargs:
            args.extend(["--lambda_dff", str(kwargs['lambda_dff'])])
        if 'cf_steps' in kwargs:
            args.extend(["--cf_steps", str(kwargs['cf_steps'])])
        if 'cf_lr' in kwargs:
            args.extend(["--cf_lr", str(kwargs['cf_lr'])])
        if 'cf_tv' in kwargs:
            args.extend(["--cf_tv", str(kwargs['cf_tv'])])
        if 'cf_every_n' in kwargs:
            args.extend(["--cf_every_n", str(kwargs['cf_every_n'])])
        if 'cf_downsample' in kwargs:
            args.extend(["--cf_downsample", str(kwargs['cf_downsample'])])
        return args


def main():
    """Main execution function with DFF (name only) Steering Counterfactual validation."""

    # Define DFF (name only) Steering Counterfactual arguments
    dff_args = {
        'steering_model_path': {
            'type': str,
            'required': True,
            'help': 'Path to steering prediction model'
        },
        'beta': {
            'type': float,
            'default': 0.3,
            'help': 'OVF steering output consistency loss weight (0.25-0.35 recommended)'
        },
        'lambda_dff': {
            'type': float,
            'default': 0.08,
            'help': 'DFF (name only) counterfactual changemap consistency weight (0.06-0.1 recommended)'
        },
        'cf_steps': {
            'type': int,
            'default': 80,
            'help': 'DFF (name only) counterfactual inner steps (reduce for speed)'
        },
        'cf_lr': {
            'type': float,
            'default': 3e-2,
            'help': 'DFF (name only) counterfactual step size'
        },
        'cf_tv': {
            'type': float,
            'default': 1e-3,
            'help': 'DFF (name only) counterfactual TV strength'
        },
        'cf_every_n': {
            'type': int,
            'default': 1,
            'help': 'Compute DFF (name only) every N steps (>=1)'
        },
        'cf_downsample': {
            'type': int,
            'default': 16,
            'help': 'Downsample DFF (name only) maps to NxN before loss'
        }
    }

    # Parse arguments
    args, parser = create_robust_argument_parser("dff_steering_counterfactual", dff_args)

    # Create trainer instance
    trainer = VKITTIDFFCounterfactualSteeringTrainer()

    # Execute training
    success = trainer.run_training(
        vkitti_seg_path=args.vkitti_seg_path,
        kitti_images_path=args.kitti_images_path,
        output_dir=args.output_dir,
        steps=args.steps,
        steering_model_path=args.steering_model_path,
        beta=args.beta,
        lambda_dff=args.lambda_dff,
        cf_steps=args.cf_steps,
        cf_lr=args.cf_lr,
        cf_tv=args.cf_tv,
        cf_every_n=args.cf_every_n,
        cf_downsample=args.cf_downsample,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path
    )

    if success:
        trainer.logger.info("🎉 V-KITTI + DFF (name only) Steering Counterfactual training completed successfully!")
        sys.exit(0)
    else:
        trainer.logger.error("💥 V-KITTI + DFF (name only) Steering Counterfactual training failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
