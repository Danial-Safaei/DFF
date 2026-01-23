#!/usr/bin/env python3
"""
V-KITTI SEGMENTATION + OVF YOLOP LL - CORRECTED VERSION

Robust V-KITTI class segmentation + OVF YOLOP lane line consistency training with
CORRECTED model loading, output handling, and import management based on
working visualization code.

CORRECTIONS APPLIED:
- Fixed YOLOP model loading with proper get_net() import
- Improved import path management for YOLOP library
- Added robust to_prob_mask() for output handling
- Corrected tensor operations and device management
- Lane line specific output handling (index 2)

Usage:
    python train_k2vk_controlnet_vkitti_seg_ovf_yolop_ll_CORRECTED.py \
        --vkitti_seg_path /path/to/vkitti/segmentation \
        --kitti_images_path /path/to/kitti/images \
        --output_dir /path/to/output \
        --yolop_weights_path /path/to/End-to-end.pth \
        --beta 0.4 \
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


class VKITTIOVFYolopLLCorrected(VKITTITrainingBase):
    """
    V-KITTI + OVF YOLOP LL trainer with CORRECTED architecture.
    """

    def __init__(self):
        super().__init__("ovf_yolop_ll_corrected")

    def validate_model_paths(self, **kwargs) -> bool:
        """Validate OVF YOLOP LL-specific model paths."""
        yolop_weights_path = kwargs.get('yolop_weights_path')
        if not yolop_weights_path:
            self.logger.error("❌ YOLOP weights path required for OVF training")
            return False

        if not os.path.exists(yolop_weights_path):
            self.logger.error(f"❌ YOLOP weights not found: {yolop_weights_path}")
            return False

        self.logger.info(f"✅ YOLOP weights validated: {yolop_weights_path}")
        return True

    def apply_variant_modifications(self, base_content: str, **kwargs) -> str:
        self.logger.info("🔧 Applying V-KITTI + OVF YOLOP LL CORRECTED modifications...")

        # CRITICAL: Marker-based patching must fail fast.
        def must_contain(content: str, marker: str, label: str) -> None:
            if marker not in content:
                raise RuntimeError(f"Required patch marker not found ({label}): {marker!r}")

        beta = kwargs.get('beta', 0.4)
        yolop_root = kwargs.get('yolop_root', '/networkhome/WMGDS/safaei_d/3_generators/models/pretrained/YOLOP')
        yolop_weights_path = kwargs.get('yolop_weights_path', '/networkhome/WMGDS/safaei_d/3_generators/models/pretrained/YOLOP/weights/End-to-end.pth')
        modifications = []

        import_marker = "import diffusers"
        must_contain(base_content, import_marker, "diffusers import")
        ovf_imports = f'''import diffusers

# V-KITTI + OVF YOLOP LL CORRECTED Imports
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import sys, os
from pathlib import Path

# CORRECTED YOLOP import path setup (based on working visualization)
YOLOP_ROOT = "{yolop_root}"
if YOLOP_ROOT not in sys.path:
    sys.path.append(YOLOP_ROOT)  # so 'import lib.*' works
lib_path = os.path.join(YOLOP_ROOT, "lib")
if lib_path not in sys.path:
    sys.path.append(lib_path)   # so 'from models.*' works

# V-KITTI Class Mapping
VKITTI_CLASS_MAPPING = {{
    'Terrain': (210, 0, 200, 0), 'Sky': (90, 200, 255, 1), 'Tree': (0, 199, 0, 2),
    'Vegetation': (90, 240, 0, 3), 'Building': (140, 140, 140, 4), 'Road': (100, 60, 100, 5),
    'GuardRail': (250, 100, 255, 6), 'TrafficSign': (255, 255, 0, 7), 'TrafficLight': (200, 200, 0, 8),
    'Pole': (255, 130, 0, 9), 'Misc': (80, 80, 80, 10), 'Truck': (160, 60, 60, 11),
    'Car': (255, 127, 80, 12), 'Van': (0, 139, 139, 13), 'Undefined': (0, 0, 0, 14)
}}
RGB_TO_INDEX = {{(r, g, b): idx for _, (r, g, b, idx) in VKITTI_CLASS_MAPPING.items()}}

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

def to_prob_mask_corrected(seg_output):
    """Handle YOLOP outputs - CORRECTED version from working visualization"""
    if not isinstance(seg_output, torch.Tensor):
        seg_output = torch.as_tensor(seg_output)
    if seg_output.dim() == 4:
        b, c, h, w = seg_output.shape
        if c == 2:
            probs = torch.softmax(seg_output, dim=1)[:, 1]  # foreground channel
        elif c == 1:
            probs = torch.sigmoid(seg_output)[:, 0]
        else:
            # Best-effort: take max probability across channels
            probs = torch.softmax(seg_output, dim=1).max(dim=1).values
    elif seg_output.dim() == 3:
        # Assume already [B,H,W] logits
        probs = torch.sigmoid(seg_output)
    else:
        raise ValueError(f"Unexpected seg_output shape: {{tuple(seg_output.shape)}}")
    return probs.clamp(0, 1)

def load_yolop_model_corrected(yolop_root, weights_path, device):
    """Load YOLOP model with CORRECTED approach (based on working visualization)"""
    try:
        # Use the builder provided by YOLOP.py: get_net(cfg) -> MCnet(YOLOP)
        try:
            from lib.models.YOLOP import get_net
        except Exception:
            from models.YOLOP import get_net

        model = get_net(False)  # cfg not used in this implementation; returns MCnet(YOLOP)
        if os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location='cpu')
            if isinstance(checkpoint, dict):
                # Accept nested 'state_dict' if present
                state = checkpoint.get('state_dict', checkpoint)
                model.load_state_dict(state, strict=False)
            else:
                # Fallback: try direct load
                model.load_state_dict(checkpoint, strict=False)

        model = model.to(device).eval()
        print(f"✅ YOLOP model loaded successfully from {{weights_path}}")
        return model
    except Exception as e:
        print(f"⚠️ YOLOP model failed to load: {{e}}")
        return None

# ---- Robust tensor extraction helpers (from corrected steering script) ----
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
    """Compute an x0 estimate in a differentiable way.

    Why: some schedulers implement `.step()` under `torch.no_grad()`.
    If OVF uses `.step()` directly, gradients may not reach the generator.
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

YOLOP_MODEL = None
OVF_BETA = {beta}
YOLOP_WEIGHTS_PATH = "{yolop_weights_path}"
'''
        modifications.append((import_marker, ovf_imports))

        args_marker = "if args.seed is not None:"
        must_contain(base_content, args_marker, "seed block")
        ovf_setup = f'''    # OVF YOLOP LL Model Setup (CORRECTED)
    global YOLOP_MODEL, OVF_BETA
    # IMPORTANT: If beta>0, OVF MUST be active; do not silently skip (that would revert to baseline behavior).
    requested_beta = float(getattr(args, 'beta', {beta}))
    yolop_weights_path = getattr(args, 'yolop_weights_path', YOLOP_WEIGHTS_PATH)
    yolop_root = getattr(args, 'yolop_root', YOLOP_ROOT)

    if yolop_weights_path and os.path.exists(yolop_weights_path):
        try:
            YOLOP_MODEL = load_yolop_model_corrected(yolop_root, yolop_weights_path, accelerator.device)
            if YOLOP_MODEL is None or getattr(YOLOP_MODEL, "is_dummy", False):
                raise RuntimeError("YOLOP model loader returned a dummy/fallback model")
            if YOLOP_MODEL is not None:
                try:
                    for p in YOLOP_MODEL.parameters():
                        p.requires_grad = False
                    # Keep external head in fp32 for numerical/stability; avoid autocast/bias dtype mismatches
                    YOLOP_MODEL = YOLOP_MODEL.to(accelerator.device).float()
                    YOLOP_MODEL.eval()
                    print("✅ YOLOP model frozen (no param updates); gradients flow to inputs.")
                except Exception:
                    pass
            print(f"✅ YOLOP LL model loaded via CORRECTED loader: {{yolop_weights_path}}")
        except Exception as e:
            # NOTE: this wrapper builds a *generated* Python script using an f-string template.
            # Avoid `{...}` in this template unless you really mean to interpolate at generation time.
            raise RuntimeError(f"CORRECTED YOLOP LL loading failed: {{e}}")
    else:
        # Strict mode: if user asked for OVF (beta>0) we must not proceed with baseline-only training.
        msg = f"YOLOP weights path invalid: {{yolop_weights_path}}"
        if requested_beta > 0:
            raise RuntimeError(msg + " (beta>0 so OVF cannot be skipped)")
        print(f"⚠️ {{msg}} - beta==0 so OVF is disabled.")

    OVF_BETA = requested_beta
    print(f"🎯 OVF Beta (YOLOP LL loss weight): {{OVF_BETA}}")

    if args.seed is not None:'''
        modifications.append((args_marker, ovf_setup))

        dataset_prep_marker = "train_dataset = prepare_train_dataset(train_dataset, accelerator)"
        must_contain(base_content, dataset_prep_marker, "dataset prep")
        ovf_preproc = '''    # V-KITTI + OVF YOLOP LL Segmentation Preprocessing (ALIGNED WITH BASELINE)
    from datasets import Image as DatasetImage
    train_dataset = train_dataset.cast_column(args.image_column, DatasetImage())
    train_dataset = train_dataset.cast_column(args.conditioning_image_column, DatasetImage())

    def preprocess_vkitti_segmentation_ovf_yolop_ll(examples):
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

    train_dataset = train_dataset.map(preprocess_vkitti_segmentation_ovf_yolop_ll, batched=True)
    print(f"✅ Applied V-KITTI+OVF YOLOP LL preprocessing (BASELINE-ALIGNED) to {{len(train_dataset)}} samples")

    train_dataset = prepare_train_dataset(train_dataset, accelerator)'''
        modifications.append((dataset_prep_marker, ovf_preproc))

        loss_marker = 'loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")'
        ovf_loss = '''                # OVF YOLOP LL Consistency (GRADIENT-ENABLED)
                reconstruction_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss = reconstruction_loss

                if YOLOP_MODEL is not None and OVF_BETA > 0:
                    try:
                        # Get real and predicted images (keep gradient path for predicted)
                        x_tgt_normalized = batch["pixel_values"]
                        # Differentiable x0 reconstruction is critical for OVF to affect the generator.
                        pred_original_sample = _predict_x0_from_model_pred(noise_scheduler, model_pred, timesteps, noisy_latents)
                        decoded_output = _safe_vae_decode(vae, pred_original_sample)
                        x_tgt_pred_normalized = decoded_output[0] if isinstance(decoded_output, tuple) else decoded_output

                        # Normalize to [0,1]
                        x_tgt = (x_tgt_normalized * 0.5) + 0.5
                        x_tgt_pred = (x_tgt_pred_normalized * 0.5) + 0.5

                        # Resize for YOLOP model input (640x384)
                        x_ll = F.interpolate(x_tgt, size=(384, 640), mode='bilinear', align_corners=False)
                        x_ll_pred = F.interpolate(x_tgt_pred, size=(384, 640), mode='bilinear', align_corners=False)

                        # Forward pass: force fp32 by disabling autocast (avoid bf16/fp16 input vs fp32 bias errors)
                        with torch.no_grad():
                            with torch.autocast("cuda", enabled=False):
                                yolop_outputs_real = YOLOP_MODEL(x_ll.float())
                        with torch.autocast("cuda", enabled=False):
                            yolop_outputs_pred = YOLOP_MODEL(x_ll_pred.float())

                        # Extract LL outputs (index 2 preferred)
                        if isinstance(yolop_outputs_real, (list, tuple)) and len(yolop_outputs_real) >= 3:
                            ll_output_real = yolop_outputs_real[2]
                        elif isinstance(yolop_outputs_real, (list, tuple)) and len(yolop_outputs_real) >= 2:
                            ll_output_real = yolop_outputs_real[-1]
                        else:
                            ll_output_real = yolop_outputs_real

                        if isinstance(yolop_outputs_pred, (list, tuple)) and len(yolop_outputs_pred) >= 3:
                            ll_output_pred = yolop_outputs_pred[2]
                        elif isinstance(yolop_outputs_pred, (list, tuple)) and len(yolop_outputs_pred) >= 2:
                            ll_output_pred = yolop_outputs_pred[-1]
                        else:
                            ll_output_pred = yolop_outputs_pred

                        # Convert to probability masks (differentiable)
                        ll_probs_real = to_prob_mask_corrected(ll_output_real)
                        ll_probs_pred = to_prob_mask_corrected(ll_output_pred)

                        # OVF loss
                        loss_ovf_ll = F.mse_loss(ll_probs_real, ll_probs_pred)

                        # Optional proof-of-gradient (disabled by default). Enable with DEBUG_GRAD_CHECK=1.
                        if os.environ.get("DEBUG_GRAD_CHECK", "0").lower() in ("1", "true", "yes"):
                            try:
                                g = torch.autograd.grad(loss_ovf_ll, model_pred, retain_graph=True, allow_unused=True)[0]
                                if g is None:
                                    print("[OVF][yolop_ll] grad(model_pred) is None (OVF may not affect generator)")
                                else:
                                    gn = float(g.detach().norm().item())
                                    if accelerator.is_local_main_process:
                                        print(f"[OVF][yolop_ll] grad_norm(model_pred)={gn:.6g}")
                            except Exception as _ge:
                                if accelerator.is_local_main_process:
                                    print(f"[OVF][yolop_ll] grad check failed: {_ge}")

                        # Adaptive weighting based on lane content
                        lane_weight = (batch["conditioning_pixel_values"] > 0.7).float().mean().item()
                        adaptive_beta = OVF_BETA * (1.0 + 0.3 * lane_weight)

                        loss = reconstruction_loss + adaptive_beta * loss_ovf_ll

                        if accelerator.is_local_main_process:
                            accelerator.log({
                                "reconstruction_loss": reconstruction_loss.detach().item(),
                                "ovf_yolop_ll_loss": loss_ovf_ll.detach().item(),
                                "adaptive_beta": adaptive_beta,
                                "lane_weight": lane_weight,
                                "total_loss": loss.detach().item(),
                            }, step=global_step)

                    except Exception as e:
                        if accelerator.is_local_main_process:
                            print(f"⚠️ OVF YOLOP LL calculation failed, using reconstruction only: {e}")
                        loss = reconstruction_loss'''
        modifications.append((loss_marker, ovf_loss))

        # Add OVF-specific arguments to parser
        args_parser_marker = '    if input_args is not None:'
        ovf_args_addition = '''    # OVF YOLOP LL Arguments (CORRECTED)
    parser.add_argument("--yolop_weights_path", type=str, required=False,
                       default="/networkhome/WMGDS/safaei_d/3_generators/models/pretrained/YOLOP/weights/End-to-end.pth",
                       help="Path to YOLOP model weights")
    parser.add_argument("--yolop_root", type=str, required=False,
                       default="/networkhome/WMGDS/safaei_d/3_generators/models/pretrained/YOLOP",
                       help="Path to YOLOP repository root")
    parser.add_argument("--beta", type=float, default=0.4, help="OVF YOLOP LL loss weight")

    if input_args is not None:'''
        modifications.append((args_parser_marker, ovf_args_addition))

        modified_content = base_content
        for marker, repl in modifications:
            if marker in modified_content:
                modified_content = modified_content.replace(marker, repl)
            else:
                raise RuntimeError(f"Required patch marker not found during OVF YOLOP LL CORRECTED: {marker[:120]}")
        # Cleanup escaped double braces in template (convert to single braces for valid Python)
        modified_content = modified_content.replace("{{", "{").replace("}}", "}")
        return modified_content

    def get_training_arguments(self, **kwargs) -> list:
        args = []
        if 'yolop_weights_path' in kwargs:
            args.extend(["--yolop_weights_path", kwargs['yolop_weights_path']])
        if 'yolop_root' in kwargs:
            args.extend(["--yolop_root", kwargs['yolop_root']])
        if 'beta' in kwargs:
            args.extend(["--beta", str(kwargs['beta'])])
        return args


def main():
    """Main execution function with OVF YOLOP LL CORRECTED validation."""

    # Define OVF YOLOP LL CORRECTED-specific arguments
    ovf_args = {
        'yolop_weights_path': {
            'type': str,
            'required': False,
            'default': '/networkhome/WMGDS/safaei_d/3_generators/models/pretrained/YOLOP/weights/End-to-end.pth',
            'help': 'Path to YOLOP model weights'
        },
        'yolop_root': {
            'type': str,
            'required': False,
            'default': '/networkhome/WMGDS/safaei_d/3_generators/models/pretrained/YOLOP',
            'help': 'Path to YOLOP repository root'
        },
        'beta': {
            'type': float,
            'default': 0.4,
            'help': 'OVF YOLOP LL loss weight (0.35-0.45 recommended for high effect)'
        }
    }

    # Parse arguments with OVF validation
    args, parser = create_robust_argument_parser("ovf_yolop_ll_corrected", ovf_args)

    # Create trainer instance
    trainer = VKITTIOVFYolopLLCorrected()

    # Execute training with OVF parameters
    success = trainer.run_training(
        vkitti_seg_path=args.vkitti_seg_path,
        kitti_images_path=args.kitti_images_path,
        output_dir=args.output_dir,
        steps=args.steps,
        yolop_weights_path=args.yolop_weights_path,
        yolop_root=args.yolop_root,
        beta=args.beta,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path
    )

    if success:
        trainer.logger.info("🎉 V-KITTI + OVF YOLOP LL CORRECTED training completed successfully!")
        sys.exit(0)
    else:
        trainer.logger.error("💥 V-KITTI + OVF YOLOP LL CORRECTED training failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
