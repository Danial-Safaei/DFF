#!/usr/bin/env python3
"""
V-KITTI SEGMENTATION + OVF STEERING - CORRECTED VERSION

Robust V-KITTI class segmentation + OVF steering consistency training with
CORRECTED model loading, output handling, and import management based on
working visualization code.

CORRECTIONS APPLIED:
- Fixed steering model loading with proper dynamic import
- Improved import path management
- Robust tensor handling and error management
- Corrected steering model architecture path handling

Usage:
    python train_k2vk_controlnet_vkitti_seg_ovf_steering_CORRECTED.py \
        --vkitti_seg_path /path/to/vkitti/segmentation \
        --kitti_images_path /path/to/kitti/images \
        --output_dir /path/to/output \
        --steering_arch_path /path/to/steering_architecture.py \
        --steering_weights_path /path/to/steering_weights.pth \
        --beta 0.6 \
        --steps 50000
"""

import os
import sys
import pathlib

# Allow this wrapper to live outside the core vkitti_seg module
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_VKITTI_SEG_DIR = _THIS_DIR.parent / "vkitti_seg"
if str(_VKITTI_SEG_DIR) not in sys.path:
    sys.path.insert(0, str(_VKITTI_SEG_DIR))

from vkitti_training_base import VKITTITrainingBase, create_robust_argument_parser


class VKITTIOVFSteeringTrainerCorrected(VKITTITrainingBase):
    """
    V-KITTI + OVF Steering trainer with CORRECTED architecture.
    """

    def __init__(self):
        super().__init__("ovf_steering_corrected")

    def validate_model_paths(self, **kwargs) -> bool:
        """Validate OVF-specific model paths."""
        steering_arch_path = kwargs.get('steering_arch_path')
        steering_weights_path = kwargs.get('steering_weights_path')
        if not steering_arch_path:
            self.logger.error("❌ Steering architecture path required for OVF training")
            return False

        if not steering_weights_path:
            self.logger.error("❌ Steering weights path required for OVF training")
            return False

        if not os.path.exists(steering_arch_path):
            self.logger.error(f"❌ Steering architecture not found: {steering_arch_path}")
            return False

        if not os.path.exists(steering_weights_path):
            self.logger.error(f"❌ Steering weights not found: {steering_weights_path}")
            return False

        self.logger.info(f"✅ Steering architecture validated: {steering_arch_path}")
        self.logger.info(f"✅ Steering weights validated: {steering_weights_path}")
        return True

    def apply_variant_modifications(self, base_content: str, **kwargs) -> str:
        self.logger.info("🔧 Applying V-KITTI + OVF Steering CORRECTED modifications...")

        # CRITICAL: Marker-based patching must fail fast.
        # If the upstream HuggingFace script changes, abort instead of silently skipping OVF logic.
        def must_contain(content: str, marker: str, label: str) -> None:
            if marker not in content:
                raise RuntimeError(f"Required patch marker not found ({label}): {marker!r}")

        beta = kwargs.get('beta', 0.6)
        default_arch = _THIS_DIR.parent / "models" / "architectures" / "For_steering_angle_prediction_tweaked_v3.py"
        default_weights = _THIS_DIR.parent / "models" / "weights" / "final_model.pth"
        steering_arch_path = kwargs.get('steering_arch_path', str(default_arch))
        steering_weights_path = kwargs.get('steering_weights_path', str(default_weights))
        modifications = []

        # 1) Import block insertion with robust helpers
        import_marker = "import diffusers"
        ovf_imports_template = '''import diffusers

# V-KITTI + OVF Steering CORRECTED Imports
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import sys, os
from pathlib import Path

# Use shared, defensive loader (handles state_dict quirks)
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
    if hasattr(rgb_seg_map, 'convert'):
        rgb_seg_map = np.array(rgb_seg_map)
    h, w = rgb_seg_map.shape[:2]
    class_indices = np.zeros((h, w), dtype=np.uint8)
    for (r, g, b), class_idx in RGB_TO_INDEX.items():
        mask = ((np.abs(rgb_seg_map[:, :, 0] - r) <= 2) &
                (np.abs(rgb_seg_map[:, :, 1] - g) <= 2) &
                (np.abs(rgb_seg_map[:, :, 2] - b) <= 2))
        class_indices[mask] = class_idx
    return class_indices

STEERING_MODEL = None
OVF_BETA = __OVF_BETA__
STEERING_ARCH_PATH = "__STEERING_ARCH_PATH__"
STEERING_WEIGHTS_PATH = "__STEERING_WEIGHTS_PATH__"

# ---- Helpers to robustly extract tensors from various scheduler/decoder outputs ----
def _extract_first_tensor(obj):
    """Recursively extract the first torch.Tensor from nested outputs."""
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
    """Return a best-effort pred_original_sample (x0) tensor from the scheduler step output."""
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
    """Compute an $x_0$ estimate in a differentiable way.

    Why: many diffusers schedulers implement `.step()` under `torch.no_grad()` (inference-oriented).
    If we used `.step()` here, OVF/DFF could become a *no-op* for the generator (no gradients flow).

    This uses the standard DDPM identities:
    - epsilon prediction: x0 = (x_t - sqrt(1-a)*eps) / sqrt(a)
    - v prediction:      x0 = sqrt(a)*x_t - sqrt(1-a)*v
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
    # Fallback (may be non-differentiable depending on scheduler implementation)
    return _safe_scheduler_pred_x0(noise_scheduler, model_pred, timesteps, noisy_latents)


def _safe_vae_decode(vae, latents):
    """Decode latents -> image tensor, handling DecoderOutput/dict/tuple formats robustly.

    IMPORTANT: decode in fp32 to avoid bf16 input vs fp32 bias errors inside VAE convs.
    """
    latents_f32 = latents.float()
    decoded = vae.decode(latents_f32 / vae.config.scaling_factor)
    t = _extract_first_tensor(decoded)
    if t is None:
        decoded2 = vae.decode(latents_f32 / vae.config.scaling_factor, return_dict=False)
        t = _extract_first_tensor(decoded2)
    if t is None:
        raise RuntimeError("VAE decode produced no tensor output")
    return t
'''
        ovf_imports = ovf_imports_template.replace("__OVF_BETA__", str(beta)).replace(
            "__STEERING_ARCH_PATH__", steering_arch_path
    ).replace("__STEERING_WEIGHTS_PATH__", steering_weights_path)
        modifications.append((import_marker, ovf_imports))

        # 2) OVF setup before seed handling
        args_marker = "if args.seed is not None:"
        ovf_setup_template = '''    # OVF Steering Model Setup (CORRECTED with proper loading)
    global STEERING_MODEL, OVF_BETA
    # IMPORTANT: If beta>0, OVF MUST be active; do not silently skip (that would revert to baseline behavior).
    requested_beta = float(getattr(args, 'beta', __DEFAULT_BETA__))
    steering_arch_path = getattr(args, 'steering_arch_path', STEERING_ARCH_PATH)
    steering_weights_path = getattr(args, 'steering_weights_path', STEERING_WEIGHTS_PATH)
    if steering_arch_path and os.path.exists(steering_arch_path) and steering_weights_path and os.path.exists(steering_weights_path):
        try:
            STEERING_MODEL = load_steering_model(steering_arch_path, steering_weights_path, accelerator.device)
            if STEERING_MODEL is None or getattr(STEERING_MODEL, "is_dummy", False):
                raise RuntimeError("Steering model loader returned a dummy/fallback model")
            if STEERING_MODEL is not None:
                # Freeze external head; allow gradients to flow to inputs
                try:
                    for p in STEERING_MODEL.parameters():
                        p.requires_grad = False
                    # Keep external head in fp32 for numerical/stability; avoid autocast/bias dtype mismatches
                    STEERING_MODEL = STEERING_MODEL.to(accelerator.device).float()
                    STEERING_MODEL.eval()
                    print("✅ Steering model frozen (no param updates); gradients flow to inputs.")
                except Exception:
                    pass
            print(f"✅ Steering model loaded: arch={steering_arch_path} weights={steering_weights_path}")
        except Exception as e:
            raise RuntimeError(f"CORRECTED steering loading failed: {e}")
    else:
        # Strict mode: if user asked for OVF (beta>0) we must not proceed with baseline-only training.
        msg = f"Steering arch/weights invalid: arch={steering_arch_path} weights={steering_weights_path}"
        if requested_beta > 0:
            raise RuntimeError(msg + " (beta>0 so OVF cannot be skipped)")
        print(f"⚠️ {msg} - beta==0 so OVF is disabled.")

    OVF_BETA = requested_beta
    print(f"🎯 OVF Beta (steering loss weight): {OVF_BETA}")

    if args.seed is not None:'''
        ovf_setup = ovf_setup_template.replace("__DEFAULT_BETA__", str(beta))
        modifications.append((args_marker, ovf_setup))

        # 3) Dataset preprocessing (ensure conditioning images are resized and cast correctly)
        dataset_prep_marker = "train_dataset = prepare_train_dataset(train_dataset, accelerator)"
        must_contain(base_content, dataset_prep_marker, "dataset prep")
        ovf_preproc = '''    # V-KITTI + OVF Segmentation Preprocessing (ALIGNED WITH BASELINE)
    from datasets import Image as DatasetImage
    train_dataset = train_dataset.cast_column(args.image_column, DatasetImage())
    train_dataset = train_dataset.cast_column(args.conditioning_image_column, DatasetImage())

    def preprocess_vkitti_segmentation_ovf_steering(examples):
        conditioning_images = []
        for seg_path in examples[args.conditioning_image_column]:
            if isinstance(seg_path, str):
                seg_img = Image.open(seg_path).convert('RGB')
            else:
                seg_img = seg_path.convert('RGB')

            # Resize segmentation conditioning to match training resolution (no hardcoded KITTI size)
            seg_resized = seg_img.resize((args.resolution_width, args.resolution_height), Image.NEAREST)

            # PROPER APPROACH: Pure segmentation as ControlNet conditioning (MATCHES BASELINE)
            conditioning_images.append(seg_resized)

        examples[args.conditioning_image_column] = conditioning_images
        return examples

    train_dataset = train_dataset.map(preprocess_vkitti_segmentation_ovf_steering, batched=True)
    print(f"✅ Applied V-KITTI+OVF STEERING preprocessing (BASELINE-ALIGNED) to {len(train_dataset)} samples")

    train_dataset = prepare_train_dataset(train_dataset, accelerator)'''
        modifications.append((dataset_prep_marker, ovf_preproc))

        # 4) Loss modification: add OVF steering consistency term
        loss_marker = 'loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")'
        ovf_loss = '''                # OVF Steering Consistency (GRADIENT-ENABLED)
                reconstruction_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss = reconstruction_loss

                if STEERING_MODEL is not None and OVF_BETA > 0:
                    try:
                        # Real and predicted images using safe helpers (keep gradient path for predicted)
                        x_tgt_normalized = batch["pixel_values"]
                        # Differentiable x0 reconstruction is critical for OVF to affect the generator.
                        pred_original_sample = _predict_x0_from_model_pred(noise_scheduler, model_pred, timesteps, noisy_latents)
                        x_tgt_pred_normalized = _safe_vae_decode(vae, pred_original_sample)

                        # Normalize to [0,1] for steering model
                        x_tgt = (x_tgt_normalized * 0.5) + 0.5
                        x_tgt_pred = (x_tgt_pred_normalized * 0.5) + 0.5

                        # Resize for steering model input (66x200 as in working visualization)
                        x_tgt_steer = F.interpolate(x_tgt, size=(66, 200), mode='bilinear', align_corners=False)
                        x_tgt_pred_steer = F.interpolate(x_tgt_pred, size=(66, 200), mode='bilinear', align_corners=False)

                        # Forward pass through steering model: force fp32 by disabling autocast (avoid bf16/fp16 input vs fp32 bias errors)
                        with torch.no_grad():
                            with torch.autocast("cuda", enabled=False):
                                real_steering = STEERING_MODEL(x_tgt_steer.float())
                        with torch.autocast("cuda", enabled=False):
                            pred_steering = STEERING_MODEL(x_tgt_pred_steer.float())

                        # Calculate OVF steering loss (grad flows through pred_steering → x_tgt_pred_steer → generator)
                        loss_ovf_steering = F.mse_loss(real_steering, pred_steering)

                        # Optional proof-of-gradient (disabled by default). Enable with DEBUG_GRAD_CHECK=1.
                        # This confirms OVF contributes non-zero gradient to model_pred/UNet outputs.
                        if os.environ.get("DEBUG_GRAD_CHECK", "0").lower() in ("1", "true", "yes"):
                            try:
                                g = torch.autograd.grad(loss_ovf_steering, model_pred, retain_graph=True, allow_unused=True)[0]
                                if g is None:
                                    print("[OVF][steering] grad(model_pred) is None (OVF may not affect generator)")
                                else:
                                    gn = float(g.detach().norm().item())
                                    if accelerator.is_local_main_process:
                                        print(f"[OVF][steering] grad_norm(model_pred)={gn:.6g}")
                            except Exception as _ge:
                                if accelerator.is_local_main_process:
                                    print(f"[OVF][steering] grad check failed: {_ge}")

                        # Adaptive weighting based on conditioning complexity
                        conditioning_complexity = torch.std(batch["conditioning_pixel_values"]).item()
                        adaptive_beta = OVF_BETA * (1.0 + 0.5 * conditioning_complexity)

                        # Combined loss
                        loss = reconstruction_loss + adaptive_beta * loss_ovf_steering

                        if accelerator.is_local_main_process:
                            accelerator.log({
                                "reconstruction_loss": reconstruction_loss.detach().item(),
                                "ovf_steering_loss": loss_ovf_steering.detach().item(),
                                "adaptive_beta": adaptive_beta,
                                "conditioning_complexity": conditioning_complexity,
                                "total_loss": loss.detach().item(),
                            }, step=global_step)

                    except Exception as e:
                        if accelerator.is_local_main_process:
                            try:
                                import traceback
                                traceback.print_exc()
                            except Exception:
                                pass
                            print(f"⚠️ OVF steering calculation failed, using reconstruction only: {e}")
                        loss = reconstruction_loss'''
        modifications.append((loss_marker, ovf_loss))

        # 5) CLI arguments for OVF steering
        args_parser_marker = '    if input_args is not None:'
        ovf_args_addition = '''    # OVF Steering Arguments (CORRECTED)
    parser.add_argument("--steering_arch_path", type=str, required=False,
                       default="/networkhome/WMGDS/safaei_d/3_generators/models/architectures/For_steering_angle_prediction_tweaked_v3.py",
                       help="Path to steering model architecture file")
    parser.add_argument("--steering_weights_path", type=str, required=False,
                       default="/networkhome/WMGDS/safaei_d/3_generators/models/weights/final_model.pth",
                       help="Path to steering model weights (.pth)")
    parser.add_argument("--beta", type=float, default=0.6, help="OVF steering loss weight")

    if input_args is not None:'''
        modifications.append((args_parser_marker, ovf_args_addition))

        # Apply all modifications
        modified_content = base_content
        for marker, repl in modifications:
            if marker in modified_content:
                modified_content = modified_content.replace(marker, repl)
            else:
                raise RuntimeError(f"Required patch marker not found during OVF steering CORRECTED: {marker[:120]}")
        return modified_content

    def get_training_arguments(self, **kwargs) -> list:
        args = []
        if 'steering_arch_path' in kwargs:
            args.extend(["--steering_arch_path", kwargs['steering_arch_path']])
        if 'steering_weights_path' in kwargs:
            args.extend(["--steering_weights_path", kwargs['steering_weights_path']])
        if 'beta' in kwargs:
            args.extend(["--beta", str(kwargs['beta'])])
        return args


def main():
    """Main execution function with OVF Steering CORRECTED validation."""

    # Define OVF Steering CORRECTED-specific arguments
    ovf_args = {
        'steering_arch_path': {
            'type': str,
            'required': False,
            'default': '/networkhome/WMGDS/safaei_d/3_generators/models/architectures/For_steering_angle_prediction_tweaked_v3.py',
            'help': 'Path to steering model architecture file'
        },
        'steering_weights_path': {
            'type': str,
            'required': False,
            'default': '/networkhome/WMGDS/safaei_d/3_generators/models/weights/final_model.pth',
            'help': 'Path to steering model weights (.pth)'
        },
        'beta': {
            'type': float,
            'default': 0.6,
            'help': 'OVF steering loss weight (0.5-0.7 recommended for high effect)'
        }
    }

    # Parse arguments with OVF validation
    args, parser = create_robust_argument_parser("ovf_steering_corrected", ovf_args)

    # Create trainer instance
    trainer = VKITTIOVFSteeringTrainerCorrected()

    # Execute training with OVF parameters
    success = trainer.run_training(
        vkitti_seg_path=args.vkitti_seg_path,
        kitti_images_path=args.kitti_images_path,
        output_dir=args.output_dir,
        steps=args.steps,
        steering_arch_path=args.steering_arch_path,
        steering_weights_path=args.steering_weights_path,
        beta=args.beta,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path
    )

    if success:
        trainer.logger.info("🎉 V-KITTI + OVF Steering CORRECTED training completed successfully!")
        sys.exit(0)
    else:
        trainer.logger.error("💥 V-KITTI + OVF Steering CORRECTED training failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
