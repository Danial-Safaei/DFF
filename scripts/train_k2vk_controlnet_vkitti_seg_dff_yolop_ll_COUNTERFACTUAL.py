#!/usr/bin/env python3
from pathlib import Path
import os
import sys
import pathlib

# Allow this wrapper to live outside the core vkitti_seg module
_THIS_DIR = pathlib.Path(__file__).resolve().parent
_VKITTI_SEG_DIR = _THIS_DIR.parent / "vkitti_seg"
if str(_VKITTI_SEG_DIR) not in sys.path:
    sys.path.insert(0, str(_VKITTI_SEG_DIR))

from vkitti_training_base import VKITTITrainingBase, create_robust_argument_parser


class VKITTIDFFCounterfactualYolopLLTrainer(VKITTITrainingBase):
    """V-KITTI + DFF (saliency) YOLOP LL trainer with OVF + DFF losses."""

    def __init__(self):
        super().__init__("dff_yolop_ll_counterfactual")

    def validate_model_paths(self, **kwargs) -> bool:
        yolop_path = kwargs.get('yolop_ll_model_path')
        if not yolop_path:
            self.logger.error("YOLOP LL model path required for DFF training")
            return False
        if not os.path.exists(yolop_path):
            self.logger.error("YOLOP LL model not found: " + str(yolop_path))
            return False
        self.logger.info("DFF saliency will be used; no external deps needed")
        return True

    def apply_variant_modifications(self, base_content: str, **kwargs) -> str:
        modifications = []

        # CRITICAL: Marker-based patching must fail fast.
        def must_contain(content: str, marker: str, label: str) -> None:
            if marker not in content:
                raise RuntimeError(f"Required patch marker not found ({label}): {marker!r}")

        # 1) Imports and helpers
        import_marker = "import diffusers"
        must_contain(base_content, import_marker, "diffusers import")
        dff_imports = '''import diffusers

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import os

# Enable shared loaders inside generated script.
# IMPORTANT: the generated patched script is written to ControlNet/ (see VKITTITrainingBase),
# so relative "Path(__file__).parent.parent / training/vkitti_seg" is often WRONG.
import sys as _sys_for_loader
_ABS_VKITTI_SEG = "/networkhome/WMGDS/safaei_d/3_generators/training/vkitti_seg_final7/vkitti_seg"
if _ABS_VKITTI_SEG not in _sys_for_loader.path:
    _sys_for_loader.path.insert(0, _ABS_VKITTI_SEG)
try:
    _here = Path(__file__).resolve()
    _fallback = _here.parent.parent / "vkitti_seg"
    if str(_fallback) not in _sys_for_loader.path:
        _sys_for_loader.path.append(str(_fallback))
except Exception:
    pass
from model_loading import load_yolop_branches

# Robust helpers (OVF parity)
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

    DFF/OVF must backpropagate into the generator. Many schedulers implement `.step()`
    under `torch.no_grad()` (inference), which would sever gradients.
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

# YOLOP helpers
def _first_tensor(obj):
    if isinstance(obj, torch.Tensor):
        return obj
    if isinstance(obj, (list, tuple)):
        for v in obj:
            t = _first_tensor(v)
            if t is not None:
                return t
    if isinstance(obj, dict):
        for v in obj.values():
            t = _first_tensor(v)
            if t is not None:
                return t
    return None

def _pick_spatial_tensor(obj):
    candidates = []
    def collect(o):
        if isinstance(o, torch.Tensor) and o.dim() >= 3:
            candidates.append(o)
        elif isinstance(o, (list, tuple)):
            for vv in o:
                collect(vv)
        elif isinstance(o, dict):
            for vv in o.values():
                collect(vv)
    collect(obj)
    if not candidates:
        return _first_tensor(obj)
    def score(t):
        if t.dim() >= 4:
            return int(t.shape[-1]) * int(t.shape[-2])
        if t.dim() == 3:
            return int(t.shape[-1]) * int(t.shape[-2])
        return 0
    candidates.sort(key=score, reverse=True)
    return candidates[0]

def yolop_select_ll_head(out):
    if isinstance(out, (list, tuple)) and len(out) >= 3:
        cand = out[2]
    elif isinstance(out, (list, tuple)) and len(out) >= 1:
        cand = out[-1]
    else:
        cand = out
    t = _pick_spatial_tensor(cand)
    if t is None:
        t = _pick_spatial_tensor(out)
    return t

def to_prob_mask(seg_output: torch.Tensor) -> torch.Tensor:
    if not isinstance(seg_output, torch.Tensor):
        seg_output = torch.as_tensor(seg_output)
    if seg_output.dim() == 4:
        b, c, h, w = seg_output.shape
        if c == 2:
            probs = torch.softmax(seg_output, dim=1)[:, 1]
        elif c == 1:
            probs = torch.sigmoid(seg_output)[:, 0]
        else:
            probs = torch.softmax(seg_output, dim=1).max(dim=1).values
    elif seg_output.dim() == 3:
        probs = torch.sigmoid(seg_output)
    else:
        raise ValueError("Unexpected segmentation output shape")
    return probs.clamp(0, 1)

def _yolop_prob_mean(head_tensor):
    t = head_tensor if isinstance(head_tensor, torch.Tensor) else _first_tensor(head_tensor)
    if t is None:
        raise RuntimeError("YOLOP head is not a tensor")
    probs = to_prob_mask(t)
    return probs.mean()

def _tv_loss(x, weight=1.0):
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return weight * (dx.abs().mean() + dy.abs().mean())

def _clamp01(x):
    return x.clamp(0.0, 1.0)

def yolop_ll_counterfactual(model, x_in, steps=60, lr=2e-2, tv=1e-3, enable_graph: bool = False):
    """Lane line counterfactual with truncated higher-order graph."""
    x0 = x_in if enable_graph else x_in.detach()
    x_cf = x0.clone()
    steps_eff = int(steps)
    if enable_graph:
        steps_eff = min(steps_eff, 8)
    for _ in range(max(1, steps_eff)):
        if enable_graph:
            x_cf = x_cf.clone()
            x_cf.requires_grad_(True)
        else:
            x_cf = x_cf.detach().requires_grad_(True)
        x_cf32 = x_cf.float()
        out = model(x_cf32)
        ll_head = yolop_select_ll_head(out)
        prob_mean = _yolop_prob_mean(ll_head)
        loss_inner = prob_mean + _tv_loss(x_cf32, tv)
        grad = torch.autograd.grad(loss_inner, x_cf, retain_graph=enable_graph, create_graph=enable_graph)[0]
        x_cf = (x_cf - lr * grad).clamp(0.0, 1.0)
    diff = (x_cf - x0).abs().mean(dim=1, keepdim=True)
    return (x_cf, diff) if enable_graph else (x_cf.detach(), diff.detach())

class SimpleYolopLLModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, 3, padding=1)
    def forward(self, x):
        return torch.sigmoid(self.conv(x))

YOLOP_LL_MODEL = None
OVF_BETA = 0.25
LAMBDA_DFF = 0.06
CF_LL_STEPS = 60
CF_LL_LR = 2e-2
CF_LL_TV = 1e-3
CF_EVERY_N = 1
CF_DOWNSAMPLE = 16
        '''
        modifications.append((import_marker, dff_imports))

        # 2) Setup models/args
        args_marker = "if args.seed is not None:"
        must_contain(base_content, args_marker, "seed block")
        dff_setup = '''
    # DFF YOLOP LL Model Setup
    global YOLOP_LL_MODEL, OVF_BETA, LAMBDA_DFF, CF_LL_STEPS, CF_LL_LR, CF_LL_TV, CF_EVERY_N, CF_DOWNSAMPLE

    yolop_ll_model_path = getattr(args, 'yolop_ll_model_path', None)
    if not yolop_ll_model_path or not os.path.exists(yolop_ll_model_path):
        raise RuntimeError(f"DFF YOLOP LL requires a valid --yolop_ll_model_path; got: {yolop_ll_model_path}")

    yolop_root = str(Path(__file__).parent.parent / "third_party" / "YOLOP")
    print("🔄 Initializing YOLOP LL via load_yolop_branches; root=", yolop_root, "; weights=", yolop_ll_model_path)
    # Match dtype to accelerator mixed precision to avoid (bf16/fp16) input vs fp32 bias errors
    try:
        import torch as _torch
        mp = getattr(accelerator, "mixed_precision", "no")
        _ext_dtype = _torch.bfloat16 if mp == "bf16" else (_torch.float16 if mp == "fp16" else _torch.float32)
    except Exception:
        _ext_dtype = None
    YOLOP_LL_MODEL = load_yolop_branches(yolop_root, yolop_ll_model_path, accelerator.device, require="ll", dtype=_ext_dtype)
    if YOLOP_LL_MODEL is None or getattr(YOLOP_LL_MODEL, "is_dummy", False):
        raise RuntimeError("Failed to load a real YOLOP LL branch; refusing to run DFF/OVF with a dummy/fallback model.")
    print("✅ YOLOP LL callable ready (eval mode)")

    OVF_BETA = getattr(args, 'beta', 0.25)
    LAMBDA_DFF = getattr(args, 'lambda_dff', 0.06)
    try:
        CF_LL_STEPS = int(getattr(args, 'cf_steps', CF_LL_STEPS))
        CF_LL_LR = float(getattr(args, 'cf_lr', CF_LL_LR))
        CF_LL_TV = float(getattr(args, 'cf_tv', CF_LL_TV))
        CF_EVERY_N = int(getattr(args, 'cf_every_n', CF_EVERY_N))
        CF_DOWNSAMPLE = int(getattr(args, 'cf_downsample', CF_DOWNSAMPLE))
    except Exception:
        pass
    print("OVF Beta (LL output consistency):", OVF_BETA)
    print("Lambda DFF (consistency weight):", LAMBDA_DFF)
    print("DFF (counterfactual) settings: steps=", CF_LL_STEPS, ", lr=", CF_LL_LR, ", tv=", CF_LL_TV, ", every_n=", CF_EVERY_N, ", down=", CF_DOWNSAMPLE)

    if args.seed is not None:'''
        modifications.append((args_marker, dff_setup))

        # 3) Preprocessing
        dataset_prep_marker = "train_dataset = prepare_train_dataset(train_dataset, accelerator)"
        must_contain(base_content, dataset_prep_marker, "dataset prep")
        dff_preprocessing = '''
    # V-KITTI + DFF YOLOP LL Segmentation Preprocessing
    from datasets import Image as DatasetImage
    train_dataset = train_dataset.cast_column(args.image_column, DatasetImage())
    train_dataset = train_dataset.cast_column(args.conditioning_image_column, DatasetImage())

    def preprocess_vkitti_segmentation_dff_ll_counterfactual(examples):
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

    train_dataset = train_dataset.map(preprocess_vkitti_segmentation_dff_ll_counterfactual, batched=True)
    print("Applied V-KITTI+DFF YOLOP LL segmentation preprocessing to", len(train_dataset), "samples")

    train_dataset = prepare_train_dataset(train_dataset, accelerator)'''
        modifications.append((dataset_prep_marker, dff_preprocessing))

        # 4) Loss modification
        loss_marker = 'loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")'
        dff_loss = '''
                # V-KITTI + DFF (name only) YOLOP LL Counterfactual Enhanced Loss
                reconstruction_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss = reconstruction_loss

                if YOLOP_LL_MODEL is not None and (OVF_BETA > 0 or LAMBDA_DFF > 0):
                    try:
                        x_tgt_normalized = batch["pixel_values"]
                        # Differentiable x0 reconstruction is critical for DFF/OVF to affect the generator.
                        pred_original_sample = _predict_x0_from_model_pred(noise_scheduler, model_pred, timesteps, noisy_latents)
                        decoded_output = _safe_vae_decode(vae, pred_original_sample)
                        x_tgt_pred_normalized = decoded_output[0] if isinstance(decoded_output, tuple) else decoded_output

                        x_tgt = (x_tgt_normalized * 0.5) + 0.5
                        x_tgt_pred = (x_tgt_pred_normalized * 0.5) + 0.5

                        x_tgt_y = F.interpolate(x_tgt, size=(384, 640), mode='bilinear', align_corners=False)
                        x_tgt_pred_y = F.interpolate(x_tgt_pred, size=(384, 640), mode='bilinear', align_corners=False)

                        # OVF LL consistency (real no-grad, predicted with grad)
                        try:
                            with torch.no_grad():
                                with torch.autocast("cuda", enabled=False):
                                    real_out = YOLOP_LL_MODEL(x_tgt_y.detach().float()) if callable(YOLOP_LL_MODEL) else YOLOP_LL_MODEL(x_tgt_y.detach().float())
                            with torch.autocast("cuda", enabled=False):
                                pred_out = YOLOP_LL_MODEL(x_tgt_pred_y.float()) if callable(YOLOP_LL_MODEL) else YOLOP_LL_MODEL(x_tgt_pred_y.float())
                            ll_real = yolop_select_ll_head(real_out)
                            ll_pred = yolop_select_ll_head(pred_out)
                            ll_probs_real = to_prob_mask(ll_real)
                            ll_probs_pred = to_prob_mask(ll_pred)
                            loss_ovf_ll = F.mse_loss(ll_probs_real, ll_probs_pred)
                        except Exception as ovf_e:
                            loss_ovf_ll = torch.tensor(0.0, device=accelerator.device)
                            if accelerator.is_local_main_process:
                                print("OVF LL calculation failed:", ovf_e)

                        # DFF (name only): original counterfactual change-map consistency
                        try:
                            compute_cf = True
                            try:
                                gs = int(global_step) if not isinstance(global_step, int) else global_step
                                compute_cf = (gs % max(1, CF_EVERY_N)) == 0
                            except Exception:
                                compute_cf = True

                            if compute_cf:
                                x_cf_real, diff_real = yolop_ll_counterfactual(YOLOP_LL_MODEL, x_tgt_y.detach(), steps=CF_LL_STEPS, lr=CF_LL_LR, tv=CF_LL_TV, enable_graph=False)
                                x_cf_pred, diff_pred = yolop_ll_counterfactual(YOLOP_LL_MODEL, x_tgt_pred_y, steps=CF_LL_STEPS, lr=CF_LL_LR, tv=CF_LL_TV, enable_graph=True)
                                ds = max(4, int(CF_DOWNSAMPLE))
                                diff_real_ds = F.adaptive_avg_pool2d(diff_real, (ds, ds))
                                diff_pred_ds = F.adaptive_avg_pool2d(diff_pred, (ds, ds))
                                loss_dff_cf = F.mse_loss(diff_real_ds, diff_pred_ds)

                                # Optional proof-of-gradient (disabled by default). Enable with DEBUG_GRAD_CHECK=1.
                                if os.environ.get("DEBUG_GRAD_CHECK", "0").lower() in ("1", "true", "yes"):
                                    try:
                                        g = torch.autograd.grad(loss_dff_cf, model_pred, retain_graph=True, allow_unused=True)[0]
                                        if g is None:
                                            print("[DFF][yolop_ll] grad(model_pred) is None (DFF may not affect generator)")
                                        else:
                                            gn = float(g.detach().norm().item())
                                            if accelerator.is_local_main_process:
                                                print(f"[DFF][yolop_ll] grad_norm(model_pred)={gn:.6g}")
                                    except Exception as _ge:
                                        if accelerator.is_local_main_process:
                                            print(f"[DFF][yolop_ll] grad check failed: {_ge}")
                            else:
                                loss_dff_cf = torch.tensor(0.0, device=accelerator.device)
                        except Exception as cf_e:
                            loss_dff_cf = torch.tensor(0.0, device=accelerator.device)
                            if accelerator.is_local_main_process:
                                print("DFF (counterfactual) LL calculation failed:", cf_e)

                        conditioning_edge_content = torch.var(batch["conditioning_pixel_values"]).item()
                        lane_weight = (batch["conditioning_pixel_values"] > 0.7).float().mean().item()
                        adaptive_beta = OVF_BETA * (1.0 + 0.3 * lane_weight)
                        adaptive_lambda = LAMBDA_DFF * (1.0 + 0.5 * conditioning_edge_content)

                        loss = reconstruction_loss + adaptive_beta * loss_ovf_ll + adaptive_lambda * loss_dff_cf

                        if accelerator.is_local_main_process:
                            logs = {
                                "reconstruction_loss": reconstruction_loss.detach().item(),
                                "ovf_ll_output_loss": loss_ovf_ll.detach().item(),
                                "dff_counterfactual_changemap_loss": loss_dff_cf.detach().item(),
                                "adaptive_beta": adaptive_beta,
                                "adaptive_lambda": adaptive_lambda,
                                "conditioning_edge_content": conditioning_edge_content,
                                "total_loss": loss.detach().item(),
                            }
                            accelerator.log(logs, step=global_step)

                    except Exception as e:
                        if accelerator.is_local_main_process:
                            print("DFF YOLOP LL loss failed, using reconstruction loss:", e)
                        loss = reconstruction_loss'''
        modifications.append((loss_marker, dff_loss))

        # 5) Extend parser with LL args
        args_parser_marker = '    if input_args is not None:'
        dff_args_addition = '''    # DFF YOLOP LL Counterfactual Arguments
    parser.add_argument("--yolop_ll_model_path", type=str, required=False, help="Path to YOLOP lane line model")
    parser.add_argument("--beta", type=float, default=0.25, help="OVF YOLOP LL loss weight")
    parser.add_argument("--lambda_dff", type=float, default=0.06, help="DFF counterfactual changemap consistency weight")
    parser.add_argument("--cf_steps", type=int, default=60, help="DFF (name only) counterfactual inner steps (reduce for speed)")
    parser.add_argument("--cf_lr", type=float, default=2e-2, help="DFF (name only) counterfactual step size")
    parser.add_argument("--cf_tv", type=float, default=1e-3, help="DFF (name only) counterfactual TV strength")
    parser.add_argument("--cf_every_n", type=int, default=1, help="Compute DFF (name only) every N steps (>=1)")
    parser.add_argument("--cf_downsample", type=int, default=16, help="Downsample DFF (name only) maps to NxN before loss")

    if input_args is not None:'''
        modifications.append((args_parser_marker, dff_args_addition))

        # Apply modifications
        modified_content = base_content
        for marker, replacement in modifications:
            if marker in modified_content:
                modified_content = modified_content.replace(marker, replacement)
                self.logger.info("Applied DFF (name only) YOLOP LL Counterfactual modification: " + marker[:30] + "...")
            else:
                raise RuntimeError("Required patch marker not found (DFF YOLOP LL): " + marker[:120])

        return modified_content

    def get_training_arguments(self, **kwargs) -> list:
        args = []
        if 'yolop_ll_model_path' in kwargs:
            args.extend(["--yolop_ll_model_path", kwargs['yolop_ll_model_path']])
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
    # NOTE: This wrapper must be able to call sys.exit() reliably so batch jobs
    # and shell scripts get correct exit codes after training completes.
    dff_args = {
        'yolop_ll_model_path': {
            'type': str,
            'required': True,
            'help': 'Path to YOLOP lane line prediction model'
        },
        'beta': {
            'type': float,
            'default': 0.25,
            'help': 'OVF LL output consistency loss weight (0.2-0.3 recommended)'
        },
        'lambda_dff': {
            'type': float,
            'default': 0.06,
            'help': 'DFF (name only) counterfactual changemap consistency loss weight (0.04-0.08 recommended)'
        },
        'cf_steps': {
            'type': int,
            'default': 60,
            'help': 'DFF (name only) counterfactual inner steps (reduce for speed)'
        },
        'cf_lr': {
            'type': float,
            'default': 2e-2,
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

    args, parser = create_robust_argument_parser("dff_yolop_ll_counterfactual", dff_args)

    trainer = VKITTIDFFCounterfactualYolopLLTrainer()

    success = trainer.run_training(
        vkitti_seg_path=args.vkitti_seg_path,
        kitti_images_path=args.kitti_images_path,
        output_dir=args.output_dir,
        steps=args.steps,
        yolop_ll_model_path=args.yolop_ll_model_path,
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
        trainer.logger.info("V-KITTI + DFF (name only) YOLOP LL Counterfactual training completed successfully!")
        sys.exit(0)
    else:
        trainer.logger.error("V-KITTI + DFF (name only) YOLOP LL Counterfactual training failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
