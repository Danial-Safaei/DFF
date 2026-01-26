#!/usr/bin/env python3
"""
V-KITTI SEGMENTATION + DFF STEERING + CONCEPT IMPORTANCE (Mask2Former)

Combines DFF steering training with Mask2Former-based concept importance.
Mask2Former is used as the authoritative (non-differentiable) concept source.
Concept losses gate DFF via semantic regions while preserving differentiability.

Usage:
    python train_k2vk_controlnet_vkitti_seg_dff_steering_CONCEPT.py \
        --vkitti_seg_path /path/to/vkitti/segmentation \
        --kitti_images_path /path/to/kitti/images \
        --output_dir /path/to/output \
        --steering_model_path /path/to/steering_model.pth \
        --beta 0.3 --lambda_dff 0.08 --lambda_concept 0.15 \
        --undesired_concepts "sky,vegetation,building,terrain" \
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


class VKITTIDFFSteeringConceptTrainer(VKITTITrainingBase):
    """V-KITTI + DFF Steering trainer with Mask2Former concept-importance losses."""

    def __init__(self):
        super().__init__("dff_steering_concept")

    def validate_model_paths(self, **kwargs) -> bool:
        steering_path = kwargs.get('steering_model_path')
        if not steering_path:
            self.logger.error("❌ Steering model path required for DFF+Concept training")
            return False
        if not os.path.exists(steering_path):
            self.logger.error(f"❌ Steering model not found: {steering_path}")
            return False
        self.logger.info("✅ Steering model path validated")
        return True

    def apply_variant_modifications(self, base_content: str, **kwargs) -> str:
        self.logger.info("🔧 Applying V-KITTI + DFF Steering + Concept modifications...")

        def must_contain(content: str, marker: str, label: str) -> None:
            if marker not in content:
                raise RuntimeError(f"Required patch marker not found ({label}): {marker!r}")

        beta = kwargs.get('beta', 0.3)
        lambda_dff = kwargs.get('lambda_dff', 0.08)
        lambda_concept = kwargs.get('lambda_concept', 0.15)
        undesired_concepts = kwargs.get('undesired_concepts', "sky,vegetation,building,terrain")
        undesired_scale = kwargs.get('undesired_scale', 1.0)
        mask2former_model_id = kwargs.get('mask2former_model_id', "facebook/mask2former-swin-large-cityscapes-semantic")

        modifications = []

        # 1) Imports + helpers
        import_marker = "import diffusers"
        must_contain(base_content, import_marker, "diffusers import")
        concept_imports = '''import diffusers

# V-KITTI + DFF Steering + Concept (Mask2Former) Imports
import numpy as np
 import torch
 import torch.nn.functional as F
 from PIL import Image
 import os
 import sys
 import hashlib
 import json
 from pathlib import Path
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

# Mask2Former (concept source)
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

# Make sure the generated script can import local helpers.
_ABS_VKITTI_SEG = os.environ.get("VKITTI_SEG_ROOT", "/networkhome/WMGDS/safaei_d/3_generators/training/vkitti_seg")
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

# Cityscapes 19 classes (lowercase, consistent with Mask2Former cityscapes head)
CITYSCAPES_CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle"
]
CITYSCAPES_NAME_TO_ID = {name: idx for idx, name in enumerate(CITYSCAPES_CLASS_NAMES)}

# ---- Robust helpers (scheduler, decode) ----

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
    latents_f32 = latents.float()
    decoded = vae.decode(latents_f32 / vae.config.scaling_factor)
    t = _extract_first_tensor(decoded)
    if t is None:
        decoded2 = vae.decode(latents_f32 / vae.config.scaling_factor, return_dict=False)
        t = _extract_first_tensor(decoded2)
    if t is None:
        raise RuntimeError("VAE decode produced no tensor output")
    return t

# ---- Counterfactual helpers (DFF) ----

def _tv_loss(x, weight=1.0):
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return weight * (dx.abs().mean() + dy.abs().mean())


def steering_counterfactual(model, x_in, steps=80, lr=3e-2, tv=1e-3, enable_graph: bool = False):
    x0 = x_in if enable_graph else x_in.detach()
    with torch.no_grad():
        y_orig = model(x0.float()).detach()
        target = -torch.sign(y_orig) * (y_orig.abs() + 0.5)

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
        y_pred = model(x_cf32)
        loss_inner = F.mse_loss(y_pred, target) + _tv_loss(x_cf32, tv)
        grad = torch.autograd.grad(loss_inner, x_cf, retain_graph=enable_graph, create_graph=enable_graph)[0]
        x_cf = (x_cf - lr * grad).clamp(0.0, 1.0)

    diff = (x_cf - x0).abs().mean(dim=1, keepdim=True)
    return (x_cf, diff) if enable_graph else (x_cf.detach(), diff.detach())

# ---- Mask2Former concept helpers ----

def _tensor_to_pil_list(x: torch.Tensor):
    x = x.detach().clamp(0, 1).cpu()
    return [TF.to_pil_image(x[i]) for i in range(x.shape[0])]


def _mask2former_segment_batch(processor, model, x_img: torch.Tensor, device: torch.device):
    pil_list = _tensor_to_pil_list(x_img)
    inputs = processor(images=pil_list, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        with torch.autocast("cuda", enabled=False):
            outputs = model(**inputs)
    target_sizes = [(p.height, p.width) for p in pil_list]
    seg = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
    seg = torch.stack([s.to(device) for s in seg], dim=0)
    return seg


def _image_cache_key(meta) -> str:
    try:
        if isinstance(meta, dict):
            meta = meta.get("path") or meta.get("image") or meta.get("file_name") or meta
        if isinstance(meta, (list, tuple)):
            meta = meta[0]
        if isinstance(meta, str):
            data = meta
        else:
            data = json.dumps(meta, sort_keys=True)
    except Exception:
        data = str(meta)
    return hashlib.md5(data.encode("utf-8")).hexdigest()


def _tensor_cache_key(x: torch.Tensor) -> str:
    with torch.no_grad():
        t = x.detach().clamp(0, 1)
        if t.dim() == 3:
            t = t.unsqueeze(0)
        pooled = F.adaptive_avg_pool2d(t, (8, 8)).squeeze(0)
        q = (pooled * 255).round().byte().cpu().numpy().tobytes()
    return hashlib.md5(q).hexdigest()


def _mask2former_segment_cached(processor, model, x_img: torch.Tensor, device: torch.device, cache, meta_list=None, cache_dir: str = None):
    if cache is None and not cache_dir:
        return _mask2former_segment_batch(processor, model, x_img, device)
    segs = [None] * x_img.shape[0]
    missing_idx = []
    for i in range(x_img.shape[0]):
        if meta_list is None:
            key = _tensor_cache_key(x_img[i])
        else:
            meta = meta_list[i] if isinstance(meta_list, (list, tuple)) else meta_list
            key = _image_cache_key(meta)
        cached = None
        if cache is not None and key in cache:
            cached = cache[key]
        elif cache_dir:
            try:
                path = os.path.join(cache_dir, f"{key}.pt")
                if os.path.exists(path):
                    cached = torch.load(path, map_location="cpu")
            except Exception:
                cached = None
        if cached is not None:
            segs[i] = cached
        else:
            missing_idx.append(i)
    if missing_idx:
        batch = _mask2former_segment_batch(processor, model, x_img[missing_idx], device)
        for out_idx, i in enumerate(missing_idx):
            if meta_list is None:
                key = _tensor_cache_key(x_img[i])
            else:
                meta = meta_list[i] if isinstance(meta_list, (list, tuple)) else meta_list
                key = _image_cache_key(meta)
            seg_cpu = batch[out_idx].detach().cpu()
            if cache is not None:
                cache[key] = seg_cpu
            if cache_dir:
                try:
                    tmp_path = os.path.join(cache_dir, f".tmp_{key}.pt")
                    final_path = os.path.join(cache_dir, f"{key}.pt")
                    torch.save(seg_cpu, tmp_path)
                    os.replace(tmp_path, final_path)
                except Exception:
                    pass
            segs[i] = seg_cpu
    segs = torch.stack([s.to(device) for s in segs], dim=0)
    return segs


def _resize_seg_to(seg_ids: torch.Tensor, target_hw):
    seg = seg_ids.unsqueeze(1).float()
    seg = F.interpolate(seg, size=target_hw, mode="nearest")
    return seg[:, 0].long()


def _aggregate_concept_importance(diff_map: torch.Tensor, seg_ids: torch.Tensor, num_classes: int):
    # diff_map: (B,1,H,W) or (B,H,W), seg_ids: (B,H,W)
    if diff_map.dim() == 3:
        diff = diff_map.unsqueeze(1)
    else:
        diff = diff_map
    diff = diff.abs()
    b, _, h, w = diff.shape
    seg = _resize_seg_to(seg_ids, (h, w))
    out = torch.zeros((b, num_classes), device=diff.device, dtype=diff.dtype)
    for cid in range(num_classes):
        mask = (seg == cid)
        denom = mask.sum(dim=(1, 2)).clamp(min=1)
        val = (diff[:, 0] * mask).sum(dim=(1, 2)) / denom
        out[:, cid] = val
    return out

# Globals
STEERING_MODEL = None
MASK2FORMER_PROCESSOR = None
MASK2FORMER_MODEL = None
MASK2FORMER_MODEL_ID = "__MASK2FORMER_MODEL_ID__"
OVF_BETA = __OVF_BETA__
LAMBDA_DFF = __LAMBDA_DFF__
LAMBDA_CONCEPT = __LAMBDA_CONCEPT__
CF_STEER_STEPS = 80
CF_STEER_LR = 3e-2
CF_STEER_TV = 1e-3
CF_EVERY_N = 1
CF_DOWNSAMPLE = 16
CONCEPT_EVERY_N = 1
CONCEPT_DOWNSAMPLE = 16
UNDESIRED_CONCEPTS = "__UNDESIRED_CONCEPTS__"
UNDESIRED_SCALE = __UNDESIRED_SCALE__
UNDESIRED_CONCEPT_IDS = None
MASK2FORMER_REAL_CACHE = None
MASK2FORMER_CACHE_DIR = None
CACHE_REAL_SEG = True
'''
        concept_imports = concept_imports.replace("__MASK2FORMER_MODEL_ID__", mask2former_model_id)
        concept_imports = concept_imports.replace("__OVF_BETA__", str(beta))
        concept_imports = concept_imports.replace("__LAMBDA_DFF__", str(lambda_dff))
        concept_imports = concept_imports.replace("__LAMBDA_CONCEPT__", str(lambda_concept))
        concept_imports = concept_imports.replace("__UNDESIRED_CONCEPTS__", undesired_concepts)
        concept_imports = concept_imports.replace("__UNDESIRED_SCALE__", str(undesired_scale))
        modifications.append((import_marker, concept_imports))

        # 2) Setup models/args
        args_marker = "if args.seed is not None:"
        must_contain(base_content, args_marker, "seed block")
        setup_block = '''
    # DFF Steering + Concept Setup
    global STEERING_MODEL, OVF_BETA, LAMBDA_DFF, LAMBDA_CONCEPT, MASK2FORMER_PROCESSOR, MASK2FORMER_MODEL, MASK2FORMER_MODEL_ID
    global CF_STEER_STEPS, CF_STEER_LR, CF_STEER_TV, CF_EVERY_N, CF_DOWNSAMPLE
    global CONCEPT_EVERY_N, CONCEPT_DOWNSAMPLE, UNDESIRED_CONCEPTS, UNDESIRED_SCALE, UNDESIRED_CONCEPT_IDS
    global MASK2FORMER_REAL_CACHE, MASK2FORMER_CACHE_DIR, CACHE_REAL_SEG

    steering_model_path = getattr(args, 'steering_model_path', None)
    if not steering_model_path or not os.path.exists(steering_model_path):
        raise RuntimeError(f"DFF+Concept steering requires --steering_model_path; got: {steering_model_path}")

    arch_path = getattr(args, 'steering_arch_path', None)
    if not arch_path:
        proj_arch = Path("/networkhome/WMGDS/safaei_d/3_generators/models/architectures/For_steering_angle_prediction_tweaked_v3.py")
        if proj_arch.exists():
            arch_path = str(proj_arch)
        else:
            arch_path = str(Path(__file__).parent.parent / "models" / "architectures" / "For_steering_angle_prediction_tweaked_v3.py")

    mp = getattr(accelerator, "mixed_precision", "no")
    _ext_dtype = torch.bfloat16 if mp == "bf16" else (torch.float16 if mp == "fp16" else torch.float32)
    STEERING_MODEL = load_steering_model(arch_path, steering_model_path, accelerator.device, dtype=_ext_dtype)
    if STEERING_MODEL is None or getattr(STEERING_MODEL, "is_dummy", False):
        raise RuntimeError("Failed to load real steering model")

    MASK2FORMER_MODEL_ID = getattr(args, 'mask2former_model_id', MASK2FORMER_MODEL_ID)

    # Mask2Former setup
    try:
        MASK2FORMER_PROCESSOR = AutoImageProcessor.from_pretrained(MASK2FORMER_MODEL_ID)
        MASK2FORMER_MODEL = Mask2FormerForUniversalSegmentation.from_pretrained(MASK2FORMER_MODEL_ID).to(accelerator.device)
        MASK2FORMER_MODEL.eval()
        for p in MASK2FORMER_MODEL.parameters():
            p.requires_grad = False
        print(f"✅ Mask2Former loaded: {MASK2FORMER_MODEL_ID}")
        try:
            id2label = getattr(MASK2FORMER_MODEL.config, "id2label", None)
            if isinstance(id2label, dict) and id2label:
                ordered = [id2label[k] for k in sorted(id2label.keys())]
                model_names = [str(n).replace("_", " ").lower() for n in ordered]
                if model_names != CITYSCAPES_CLASS_NAMES:
                    if accelerator.is_local_main_process:
                        print("❌ Mask2Former label map mismatch with Cityscapes order.")
                    raise RuntimeError("Mask2Former label map does not match expected Cityscapes order.")
        except Exception as _map_e:
            raise RuntimeError(f"Mask2Former label-map validation failed: {_map_e}")
    except Exception as e:
        raise RuntimeError(f"Mask2Former load failed: {e}")

    OVF_BETA = getattr(args, 'beta', OVF_BETA)
    LAMBDA_DFF = getattr(args, 'lambda_dff', LAMBDA_DFF)
    LAMBDA_CONCEPT = getattr(args, 'lambda_concept', LAMBDA_CONCEPT)
    try:
        CF_STEER_STEPS = int(getattr(args, 'cf_steps', CF_STEER_STEPS))
        CF_STEER_LR = float(getattr(args, 'cf_lr', CF_STEER_LR))
        CF_STEER_TV = float(getattr(args, 'cf_tv', CF_STEER_TV))
        CF_EVERY_N = int(getattr(args, 'cf_every_n', CF_EVERY_N))
        CF_DOWNSAMPLE = int(getattr(args, 'cf_downsample', CF_DOWNSAMPLE))
    except Exception:
        pass
    CONCEPT_EVERY_N = int(getattr(args, 'concept_every_n', CONCEPT_EVERY_N))
    CONCEPT_DOWNSAMPLE = int(getattr(args, 'concept_downsample', CONCEPT_DOWNSAMPLE))
    MASK2FORMER_CACHE_DIR = getattr(args, 'mask2former_cache_dir', None)
    CACHE_REAL_SEG = bool(int(getattr(args, 'cache_real_seg', 1))) if hasattr(args, 'cache_real_seg') else True
    MASK2FORMER_REAL_CACHE = {} if CACHE_REAL_SEG else None
    if MASK2FORMER_CACHE_DIR:
        try:
            os.makedirs(MASK2FORMER_CACHE_DIR, exist_ok=True)
        except Exception:
            MASK2FORMER_CACHE_DIR = None
    UNDESIRED_CONCEPTS = getattr(args, 'undesired_concepts', UNDESIRED_CONCEPTS)
    UNDESIRED_SCALE = float(getattr(args, 'undesired_scale', UNDESIRED_SCALE))
    try:
        _undesired = [c.strip().lower().replace("_", " ") for c in str(UNDESIRED_CONCEPTS).split(",") if c.strip()]
        _unknown = [c for c in _undesired if c not in CITYSCAPES_NAME_TO_ID]
        if _unknown and accelerator.is_local_main_process:
            print(f"⚠️ Unknown undesired concepts ignored: {_unknown}")
        UNDESIRED_CONCEPT_IDS = [CITYSCAPES_NAME_TO_ID[c] for c in _undesired if c in CITYSCAPES_NAME_TO_ID]
    except Exception:
        UNDESIRED_CONCEPT_IDS = []

    print("🎯 OVF Beta:", OVF_BETA)
    print("🎯 Lambda DFF:", LAMBDA_DFF)
    print("🎯 Lambda Concept:", LAMBDA_CONCEPT)
    print("🎯 Undesired concepts:", UNDESIRED_CONCEPTS)
    print("🎯 Mask2Former cache dir:", MASK2FORMER_CACHE_DIR or "None")
    print("🎯 Cache real seg:", CACHE_REAL_SEG)
    print("🎯 Deterministic:", getattr(args, 'deterministic', False))
    if accelerator.is_local_main_process:
        try:
            import subprocess as _subprocess
            git_hash = _subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
            print(f"🎯 Git hash: {git_hash}")
        except Exception:
            pass

    if getattr(args, 'deterministic', False):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass

    if args.seed is not None:'''
        modifications.append((args_marker, setup_block))

        # 3) Preprocessing
        dataset_prep_marker = "train_dataset = prepare_train_dataset(train_dataset, accelerator)"
        must_contain(base_content, dataset_prep_marker, "dataset prep")
        preprocessing = '''
    # V-KITTI + DFF Steering + Concept preprocessing
    from datasets import Image as DatasetImage
    train_dataset = train_dataset.cast_column(args.image_column, DatasetImage())
    train_dataset = train_dataset.cast_column(args.conditioning_image_column, DatasetImage())

    def preprocess_vkitti_segmentation_dff_concept(examples):
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

    train_dataset = train_dataset.map(preprocess_vkitti_segmentation_dff_concept, batched=True)
    print("✅ Applied V-KITTI+DFF+Concept preprocessing to", len(train_dataset), "samples")

    train_dataset = prepare_train_dataset(train_dataset, accelerator)'''
        modifications.append((dataset_prep_marker, preprocessing))

        # 4) Loss modification
        loss_marker = 'loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")'
        loss_block = '''
                # V-KITTI + DFF Steering + Concept Loss
                reconstruction_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss = reconstruction_loss

                if STEERING_MODEL is not None and (OVF_BETA > 0 or LAMBDA_DFF > 0 or LAMBDA_CONCEPT > 0):
                    try:
                        x_tgt_normalized = batch["pixel_values"]
                        pred_original_sample = _predict_x0_from_model_pred(noise_scheduler, model_pred, timesteps, noisy_latents)
                        x_tgt_pred_normalized = _safe_vae_decode(vae, pred_original_sample)

                        x_tgt = (x_tgt_normalized * 0.5) + 0.5
                        x_tgt_pred = (x_tgt_pred_normalized * 0.5) + 0.5

                        x_tgt_steer = F.interpolate(x_tgt, size=(66, 200), mode='bilinear', align_corners=False)
                        x_tgt_pred_steer = F.interpolate(x_tgt_pred, size=(66, 200), mode='bilinear', align_corners=False)

                        # OVF
                        try:
                            with torch.no_grad():
                                with torch.autocast("cuda", enabled=False):
                                    real_steer = STEERING_MODEL(x_tgt_steer.detach().float())
                            with torch.autocast("cuda", enabled=False):
                                pred_steer = STEERING_MODEL(x_tgt_pred_steer.float())
                            loss_ovf = F.mse_loss(real_steer, pred_steer)
                        except Exception as ovf_e:
                            loss_ovf = torch.tensor(0.0, device=accelerator.device)
                            if accelerator.is_local_main_process:
                                print(f"OVF steering failed: {ovf_e}")

                        # DFF (counterfactual)
                        try:
                            compute_cf = True
                            try:
                                gs = int(global_step) if not isinstance(global_step, int) else global_step
                                compute_cf = (gs % max(1, CF_EVERY_N)) == 0
                                if LAMBDA_CONCEPT > 0:
                                    compute_cf = compute_cf or ((gs % max(1, CONCEPT_EVERY_N)) == 0)
                            except Exception:
                                compute_cf = True

                            if compute_cf:
                                torch.cuda.empty_cache()
                                x_cf_real, diff_real = steering_counterfactual(STEERING_MODEL, x_tgt_steer.detach(), steps=CF_STEER_STEPS, lr=CF_STEER_LR, tv=CF_STEER_TV, enable_graph=False)
                                x_cf_pred, diff_pred = steering_counterfactual(STEERING_MODEL, x_tgt_pred_steer, steps=CF_STEER_STEPS, lr=CF_STEER_LR, tv=CF_STEER_TV, enable_graph=True)
                                ds = max(4, int(CF_DOWNSAMPLE))
                                diff_real_ds = F.adaptive_avg_pool2d(diff_real, (ds, ds))
                                diff_pred_ds = F.adaptive_avg_pool2d(diff_pred, (ds, ds))
                                loss_dff = F.mse_loss(diff_real_ds, diff_pred_ds)
                                torch.cuda.empty_cache()
                            else:
                                diff_real = None
                                diff_pred = None
                                diff_real_ds = None
                                diff_pred_ds = None
                                loss_dff = torch.tensor(0.0, device=accelerator.device)
                        except Exception as cf_e:
                            diff_real = None
                            diff_pred = None
                            diff_real_ds = None
                            diff_pred_ds = None
                            loss_dff = torch.tensor(0.0, device=accelerator.device)
                            if accelerator.is_local_main_process:
                                print(f"DFF steering failed: {cf_e}")

                        # Concept importance (Mask2Former)
                        if diff_pred is not None and os.environ.get("DEBUG_GRAD_CHECK", "0").lower() in ("1", "true", "yes"):
                            try:
                                g = torch.autograd.grad(loss_dff, model_pred, retain_graph=True, allow_unused=True)[0]
                                if g is None:
                                    print("[DFF][steering][concept] grad(model_pred) is None (DFF may not affect generator)")
                                else:
                                    gn = float(g.detach().norm().item())
                                    if accelerator.is_local_main_process:
                                        print(f"[DFF][steering][concept] grad_norm(model_pred)={gn:.6g}")
                            except Exception as grad_e:
                                if accelerator.is_local_main_process:
                                    print(f"[DFF][steering][concept] grad check failed: {grad_e}")

                        # Concept importance (Mask2Former)
                        concept_topk_logs = {}
                        loss_ci = torch.tensor(0.0, device=accelerator.device)
                        loss_ci_undesired = torch.tensor(0.0, device=accelerator.device)
                        try:
                            do_ci = True
                            try:
                                gs = int(global_step) if not isinstance(global_step, int) else global_step
                                do_ci = compute_cf and ((gs % max(1, CONCEPT_EVERY_N)) == 0)
                            except Exception:
                                do_ci = True

                            if do_ci and diff_pred is not None and diff_real is not None:
                                seg_meta = batch.get("image_path") if isinstance(batch, dict) else None
                                seg_real = _mask2former_segment_cached(MASK2FORMER_PROCESSOR, MASK2FORMER_MODEL, x_tgt, accelerator.device, MASK2FORMER_REAL_CACHE, seg_meta, MASK2FORMER_CACHE_DIR)
                                seg_pred = _mask2former_segment_batch(MASK2FORMER_PROCESSOR, MASK2FORMER_MODEL, x_tgt_pred, accelerator.device)
                                ci_ds = max(4, int(CONCEPT_DOWNSAMPLE))
                                diff_real_ci = F.adaptive_avg_pool2d(diff_real, (ci_ds, ci_ds))
                                diff_pred_ci = F.adaptive_avg_pool2d(diff_pred, (ci_ds, ci_ds))
                                ci_real = _aggregate_concept_importance(diff_real_ci, seg_real, len(CITYSCAPES_CLASS_NAMES))
                                ci_pred = _aggregate_concept_importance(diff_pred_ci, seg_pred, len(CITYSCAPES_CLASS_NAMES))
                                loss_ci = F.mse_loss(ci_pred, ci_real.detach())
                                ci_l1 = F.l1_loss(ci_pred, ci_real.detach())
                                ci_cos = F.cosine_similarity(ci_pred, ci_real.detach(), dim=1).mean()

                                undesired_ids = UNDESIRED_CONCEPT_IDS or []
                                if undesired_ids:
                                    ci_undes = ci_pred[:, undesired_ids].mean()
                                    loss_ci_undesired = ci_undes
                                try:
                                    ci_real_mean = ci_real.detach().mean(dim=0)
                                    ci_pred_mean = ci_pred.detach().mean(dim=0)
                                    k = min(5, ci_real_mean.numel())
                                    top_real_vals, top_real_idx = torch.topk(ci_real_mean, k)
                                    top_pred_vals, top_pred_idx = torch.topk(ci_pred_mean, k)
                                    for rank, (val, idx) in enumerate(zip(top_real_vals, top_real_idx), start=1):
                                        name = CITYSCAPES_CLASS_NAMES[int(idx)].replace(" ", "_")
                                        concept_topk_logs[f"ci_real_top{rank}_{name}"] = float(val.detach().item())
                                    for rank, (val, idx) in enumerate(zip(top_pred_vals, top_pred_idx), start=1):
                                        name = CITYSCAPES_CLASS_NAMES[int(idx)].replace(" ", "_")
                                        concept_topk_logs[f"ci_pred_top{rank}_{name}"] = float(val.detach().item())
                                    concept_topk_logs["concept_l1"] = float(ci_l1.detach().item())
                                    concept_topk_logs["concept_cosine"] = float(ci_cos.detach().item())
                                except Exception as topk_e:
                                    if accelerator.is_local_main_process:
                                        print(f"Concept top-k logging failed: {topk_e}")

                                if os.environ.get("DEBUG_GRAD_CHECK", "0").lower() in ("1", "true", "yes"):
                                    try:
                                        ci_total = loss_ci + UNDESIRED_SCALE * loss_ci_undesired
                                        g_ci = torch.autograd.grad(ci_total, model_pred, retain_graph=True, allow_unused=True)[0]
                                        if g_ci is None:
                                            print("[Concept][steering] grad(model_pred) is None (concept may not affect generator)")
                                        else:
                                            gn = float(g_ci.detach().norm().item())
                                            if accelerator.is_local_main_process:
                                                print(f"[Concept][steering] grad_norm(model_pred)={gn:.6g}")
                                    except Exception as grad_e:
                                        if accelerator.is_local_main_process:
                                            print(f"[Concept][steering] grad check failed: {grad_e}")
                        except Exception as ci_e:
                            if accelerator.is_local_main_process:
                                print(f"Concept importance failed: {ci_e}")

                        conditioning_complexity = torch.std(batch["conditioning_pixel_values"]).item()
                        adaptive_beta = OVF_BETA * (1.0 + 0.5 * conditioning_complexity)
                        road_content = (batch["conditioning_pixel_values"] > 0.8).float().mean().item()
                        adaptive_lambda = LAMBDA_DFF * (1.0 + 0.4 * road_content)

                        loss = (
                            reconstruction_loss
                            + adaptive_beta * loss_ovf
                            + adaptive_lambda * loss_dff
                            + LAMBDA_CONCEPT * (loss_ci + UNDESIRED_SCALE * loss_ci_undesired)
                        )

                        if accelerator.is_local_main_process:
                            accelerator.log({
                                "reconstruction_loss": reconstruction_loss.detach().item(),
                                "ovf_steering_output_loss": loss_ovf.detach().item(),
                                "dff_counterfactual_changemap_loss": loss_dff.detach().item(),
                                "concept_align_loss": loss_ci.detach().item(),
                                "concept_undesired_loss": loss_ci_undesired.detach().item(),
                                "adaptive_beta": adaptive_beta,
                                "adaptive_lambda": adaptive_lambda,
                                "lambda_concept": LAMBDA_CONCEPT,
                                "conditioning_complexity": conditioning_complexity,
                                "road_content": road_content,
                                "total_loss": loss.detach().item(),
                                **concept_topk_logs,
                            }, step=global_step)

                            if getattr(args, 'dump_concept_sample', None):
                                try:
                                    dump_path = Path(args.dump_concept_sample)
                                    dump_path.parent.mkdir(parents=True, exist_ok=True)
                                    if not dump_path.exists():
                                        payload = {
                                            "global_step": int(global_step),
                                            "concept_real_topk": {k: v for k, v in concept_topk_logs.items() if k.startswith("ci_real_top")},
                                            "concept_pred_topk": {k: v for k, v in concept_topk_logs.items() if k.startswith("ci_pred_top")},
                                            "concept_l1": concept_topk_logs.get("concept_l1"),
                                            "concept_cosine": concept_topk_logs.get("concept_cosine"),
                                            "undesired_concepts": UNDESIRED_CONCEPTS,
                                            "lambda_concept": float(LAMBDA_CONCEPT),
                                        }
                                        dump_path.write_text(json.dumps(payload, indent=2))
                                except Exception as dump_e:
                                    print(f"Concept snapshot dump failed: {dump_e}")

                    except Exception as e:
                        if accelerator.is_local_main_process:
                            print(f"DFF+Concept steering loss failed: {e}")
                        loss = reconstruction_loss'''
        modifications.append((loss_marker, loss_block))

        # 5) Extend parser args
        args_parser_marker = '    if input_args is not None:'
        args_block = '''    # DFF Steering + Concept Arguments
    parser.add_argument("--steering_model_path", type=str, required=False, help="Path to steering prediction model")
    parser.add_argument("--beta", type=float, default=0.3, help="OVF steering loss weight")
    parser.add_argument("--lambda_dff", type=float, default=0.08, help="DFF counterfactual loss weight")
    parser.add_argument("--lambda_concept", type=float, default=0.15, help="Concept importance loss weight")
    parser.add_argument("--mask2former_model_id", type=str, default="facebook/mask2former-swin-large-cityscapes-semantic", help="Mask2Former model id")
    parser.add_argument("--undesired_concepts", type=str, default="sky,vegetation,building,terrain", help="Comma-separated undesired concepts")
    parser.add_argument("--undesired_scale", type=float, default=1.0, help="Scale for undesired concept penalty")
    parser.add_argument("--mask2former_cache_dir", type=str, default=None, help="Optional cache dir for Mask2Former real segmentation")
    parser.add_argument("--cache_real_seg", type=int, default=1, help="Cache real target segmentation (1/0)")
    parser.add_argument("--concept_every_n", type=int, default=1, help="Compute concept loss every N steps")
    parser.add_argument("--concept_downsample", type=int, default=16, help="Downsample size for concept aggregation")
    parser.add_argument("--cf_steps", type=int, default=80, help="DFF counterfactual inner steps")
    parser.add_argument("--cf_lr", type=float, default=3e-2, help="DFF counterfactual step size")
    parser.add_argument("--cf_tv", type=float, default=1e-3, help="DFF counterfactual TV strength")
    parser.add_argument("--cf_every_n", type=int, default=1, help="Compute DFF every N steps")
    parser.add_argument("--cf_downsample", type=int, default=16, help="Downsample DFF maps to NxN before loss")
    parser.add_argument("--steering_arch_path", type=str, default=None, help="Path to steering architecture .py")
    parser.add_argument("--dump_concept_sample", type=str, default=None, help="Write a one-step concept snapshot JSON to this path")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic algorithms")

    if input_args is not None:''' 
        modifications.append((args_parser_marker, args_block))

        modified_content = base_content
        for marker, replacement in modifications:
            if marker in modified_content:
                modified_content = modified_content.replace(marker, replacement)
                self.logger.info(f"✅ Applied DFF+Concept modification: {marker[:30]}...")
            else:
                raise RuntimeError(f"Required patch marker not found (DFF+Concept steering): {marker[:120]}")

        return modified_content

    def get_training_arguments(self, **kwargs) -> list:
        args = []
        for key in [
            'steering_model_path', 'beta', 'lambda_dff', 'lambda_concept', 'mask2former_model_id',
            'undesired_concepts', 'undesired_scale', 'mask2former_cache_dir', 'cache_real_seg',
            'concept_every_n', 'concept_downsample', 'cf_steps', 'cf_lr', 'cf_tv', 'cf_every_n',
            'cf_downsample', 'steering_arch_path', 'dump_concept_sample', 'deterministic'
        ]:
            if key in kwargs and kwargs[key] is not None:
                if key == 'deterministic':
                    if str(kwargs[key]).lower() in ("1", "true", "yes"):
                        args.append("--deterministic")
                else:
                    args.extend([f"--{key}", str(kwargs[key])])
        return args


def main():
    dff_args = {
        'steering_model_path': {
            'type': str,
            'required': True,
            'help': 'Path to steering prediction model'
        },
        'beta': {
            'type': float,
            'default': 0.3,
            'help': 'OVF steering output consistency weight'
        },
        'lambda_dff': {
            'type': float,
            'default': 0.08,
            'help': 'DFF counterfactual loss weight'
        },
        'lambda_concept': {
            'type': float,
            'default': 0.15,
            'help': 'Concept importance loss weight'
        },
        'mask2former_cache_dir': {
            'type': str,
            'default': None,
            'help': 'Optional cache dir for Mask2Former real segmentation'
        },
        'cache_real_seg': {
            'type': int,
            'default': 1,
            'help': 'Cache real target segmentation (1/0)'
        },
        'mask2former_model_id': {
            'type': str,
            'default': 'facebook/mask2former-swin-large-cityscapes-semantic',
            'help': 'Mask2Former model id'
        },
        'undesired_concepts': {
            'type': str,
            'default': 'sky,vegetation,building,terrain',
            'help': 'Comma-separated undesired concepts'
        },
        'undesired_scale': {
            'type': float,
            'default': 1.0,
            'help': 'Scale for undesired concept penalty'
        },
        'concept_every_n': {
            'type': int,
            'default': 1,
            'help': 'Compute concept loss every N steps'
        },
        'concept_downsample': {
            'type': int,
            'default': 16,
            'help': 'Downsample size for concept aggregation'
        },
        'cf_steps': {
            'type': int,
            'default': 80,
            'help': 'DFF counterfactual inner steps'
        },
        'cf_lr': {
            'type': float,
            'default': 3e-2,
            'help': 'DFF counterfactual step size'
        },
        'cf_tv': {
            'type': float,
            'default': 1e-3,
            'help': 'DFF counterfactual TV strength'
        },
        'cf_every_n': {
            'type': int,
            'default': 1,
            'help': 'Compute DFF every N steps'
        },
        'cf_downsample': {
            'type': int,
            'default': 16,
            'help': 'Downsample DFF maps to NxN before loss'
        },
        'steering_arch_path': {
            'type': str,
            'default': None,
            'help': 'Path to steering architecture .py'
        },
        'dump_concept_sample': {
            'type': str,
            'default': None,
            'help': 'Write a one-step concept snapshot JSON to this path'
        },
        'deterministic': {
            'action': 'store_true',
            'help': 'Enable deterministic algorithms'
        }
    }

    args, parser = create_robust_argument_parser("dff_steering_concept", dff_args)

    trainer = VKITTIDFFSteeringConceptTrainer()

    success = trainer.run_training(
        vkitti_seg_path=args.vkitti_seg_path,
        kitti_images_path=args.kitti_images_path,
        output_dir=args.output_dir,
        steps=args.steps,
        steering_model_path=args.steering_model_path,
        beta=args.beta,
        lambda_dff=args.lambda_dff,
        lambda_concept=args.lambda_concept,
        mask2former_model_id=args.mask2former_model_id,
        undesired_concepts=args.undesired_concepts,
        undesired_scale=args.undesired_scale,
        mask2former_cache_dir=args.mask2former_cache_dir,
        cache_real_seg=args.cache_real_seg,
        concept_every_n=args.concept_every_n,
        concept_downsample=args.concept_downsample,
        cf_steps=args.cf_steps,
        cf_lr=args.cf_lr,
        cf_tv=args.cf_tv,
        cf_every_n=args.cf_every_n,
        cf_downsample=args.cf_downsample,
        steering_arch_path=args.steering_arch_path,
        dump_concept_sample=args.dump_concept_sample,
        deterministic=args.deterministic,
        pretrained_model_name_or_path=args.pretrained_model_name_or_path
    )

    if success:
        trainer.logger.info("🎉 V-KITTI + DFF Steering + Concept training completed successfully!")
        sys.exit(0)
    else:
        trainer.logger.error("💥 V-KITTI + DFF Steering + Concept training failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
