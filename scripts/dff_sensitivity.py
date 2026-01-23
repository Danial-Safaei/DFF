#!/usr/bin/env python
import os
import sys
import math
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "generated_figures")
REAL_IMG = os.path.join(FIG_DIR, "real_input.png")
GEN_IMG = os.path.join(FIG_DIR, "Baseline.png")

STEERING_ARCH_PATH = "/networkhome/WMGDS/safaei_d/3_generators/models/architectures/For_steering_angle_prediction_tweaked_v3.py"
STEERING_WEIGHTS_PATH = "/networkhome/WMGDS/safaei_d/3_generators/models/weights/final_model.pth"
YOLOP_ROOT = "/networkhome/WMGDS/safaei_d/3_generators/models/pretrained/YOLOP"
YOLOP_WEIGHTS_PATH = "/networkhome/WMGDS/safaei_d/3_generators/models/pretrained/YOLOP/weights/End-to-end.pth"


# ----------------- Load SUTs -----------------

def load_steering_model():
    import importlib.util
    if not os.path.exists(STEERING_ARCH_PATH):
        raise FileNotFoundError(STEERING_ARCH_PATH)
    spec = importlib.util.spec_from_file_location("steering_arch", STEERING_ARCH_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ModelClass = getattr(module, "AutonomousDrivingModel", None) or getattr(module, "SimpleSteeringModel", None)
    if ModelClass is None:
        for _, obj in module.__dict__.items():
            if isinstance(obj, type) and issubclass(obj, torch.nn.Module):
                ModelClass = obj
                break
    if ModelClass is None:
        raise RuntimeError("No steering model class found")
    model = ModelClass()
    state = torch.load(STEERING_WEIGHTS_PATH, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    def preprocess(img: Image.Image) -> torch.Tensor:
        arr = np.array(img.resize((200, 66), Image.LANCZOS)).astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        return t

    return model, preprocess


def load_yolop_model():
    if YOLOP_ROOT not in sys.path:
        sys.path.append(YOLOP_ROOT)
    lib_path = os.path.join(YOLOP_ROOT, "lib")
    if os.path.isdir(lib_path) and lib_path not in sys.path:
        sys.path.append(lib_path)
    try:
        from lib.models.YOLOP import get_net
    except Exception:
        from models.YOLOP import get_net
    model = get_net(False)
    ckpt = torch.load(YOLOP_WEIGHTS_PATH, map_location="cpu")
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    def preprocess(img: Image.Image) -> torch.Tensor:
        arr = np.array(img.resize((640, 384), Image.LANCZOS)).astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        return t

    return model, preprocess


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
    candidates.sort(key=lambda t: int(t.shape[-1]) * int(t.shape[-2]) if t.dim() >= 3 else 0, reverse=True)
    return candidates[0]


def yolop_select_da_head(out):
    cand = out[1] if isinstance(out, (list, tuple)) and len(out) > 1 else out
    t = _pick_spatial_tensor(cand)
    return t if t is not None else _pick_spatial_tensor(out)


def yolop_select_ll_head(out):
    cand = out[2] if isinstance(out, (list, tuple)) and len(out) > 2 else out
    t = _pick_spatial_tensor(cand)
    return t if t is not None else _pick_spatial_tensor(out)


def to_prob_mask(seg_output: torch.Tensor) -> torch.Tensor:
    if seg_output.dim() == 4:
        b, c, _, _ = seg_output.shape
        if c == 2:
            probs = torch.softmax(seg_output, dim=1)[:, 1]
        elif c == 1:
            probs = torch.sigmoid(seg_output)[:, 0]
        else:
            probs = torch.softmax(seg_output, dim=1).max(dim=1).values
    elif seg_output.dim() == 3:
        probs = torch.sigmoid(seg_output)
    else:
        probs = torch.sigmoid(seg_output.unsqueeze(0))
    return probs.clamp(0, 1)


def _yolop_prob_mean(head_tensor) -> torch.Tensor:
    t = head_tensor if isinstance(head_tensor, torch.Tensor) else _first_tensor(head_tensor)
    return to_prob_mask(t).mean()


def _tv_loss(x: torch.Tensor, weight: float = 1e-3) -> torch.Tensor:
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    return weight * (dx.abs().mean() + dy.abs().mean())


def _clamp01(x: torch.Tensor) -> torch.Tensor:
    return x.clamp(0.0, 1.0)


def _steering_counterfactual_robust(model, x_in: torch.Tensor, steps: int = 80, lr: float = 3e-2,
                                     tv: float = 1e-3, target_flip_threshold: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    device = x_in.device
    x_cf = x_in.detach().clone().requires_grad_(True)
    with torch.no_grad():
        y_orig = model(x_in).detach().mean()
    target_delta = target_flip_threshold * (-torch.sign(y_orig) if y_orig.abs() > 1e-6 else torch.tensor(1.0, device=device))
    target = y_orig + target_delta
    opt = torch.optim.Adam([x_cf], lr=lr)
    best_success_cf = None
    best_success_diff = float("inf")
    best_effort_cf = x_cf.detach().clone()
    best_effort_change = 0.0
    for _ in range(steps):
        opt.zero_grad()
        y_pred = model(x_cf)
        y_pred_mean = y_pred.mean()
        with torch.no_grad():
            current_change = (y_pred_mean.detach() - y_orig).abs().item()
            current_diff = (x_cf - x_in).abs().mean().item()
            if current_change > best_effort_change:
                best_effort_change = current_change
                best_effort_cf = x_cf.detach().clone()
            if current_change >= target_flip_threshold * 0.8 and current_diff < best_success_diff:
                best_success_diff = current_diff
                best_success_cf = x_cf.detach().clone()
        output_loss = F.mse_loss(y_pred_mean, target)
        perturbation_loss = (x_cf - x_in).abs().mean()
        tv_loss_val = _tv_loss(x_cf, tv)
        loss = output_loss + 0.1 * perturbation_loss + tv_loss_val
        loss.backward()
        opt.step()
        with torch.no_grad():
            x_cf.data = _clamp01(x_cf.data)
    result_cf = best_success_cf if best_success_cf is not None else best_effort_cf
    with torch.no_grad():
        diff = (result_cf - x_in).abs().mean(dim=1, keepdim=True)
    return result_cf, diff


def _yolop_counterfactual_robust(model, x_in: torch.Tensor, selector, steps: int = 80, lr: float = 2e-2,
                                  tv: float = 1e-3, target_flip_threshold: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    device = x_in.device
    x_cf = x_in.detach().clone().requires_grad_(True)
    with torch.no_grad():
        out_orig = model(x_in)
        prob_orig = _yolop_prob_mean(selector(out_orig)).item()
    target_prob = max(0.0, prob_orig - target_flip_threshold) if prob_orig > 0.5 else min(1.0, prob_orig + target_flip_threshold)
    target_tensor = torch.tensor(target_prob, device=device, dtype=x_in.dtype)
    opt = torch.optim.Adam([x_cf], lr=lr)
    best_success_cf = None
    best_success_diff = float("inf")
    best_effort_cf = x_cf.detach().clone()
    best_effort_change = 0.0
    for _ in range(steps):
        opt.zero_grad()
        out = model(x_cf)
        prob_mean = _yolop_prob_mean(selector(out))
        with torch.no_grad():
            prob_current = prob_mean.detach().item()
            current_change = abs(prob_current - prob_orig)
            current_diff = (x_cf - x_in).abs().mean().item()
            if current_change > best_effort_change:
                best_effort_change = current_change
                best_effort_cf = x_cf.detach().clone()
            if current_change >= target_flip_threshold * 0.8 and current_diff < best_success_diff:
                best_success_diff = current_diff
                best_success_cf = x_cf.detach().clone()
        output_loss = F.mse_loss(prob_mean, target_tensor)
        perturbation_loss = (x_cf - x_in).abs().mean()
        tv_loss_val = _tv_loss(x_cf, tv)
        loss = output_loss + 0.1 * perturbation_loss + tv_loss_val
        loss.backward()
        opt.step()
        with torch.no_grad():
            x_cf.data = _clamp01(x_cf.data)
    result_cf = best_success_cf if best_success_cf is not None else best_effort_cf
    with torch.no_grad():
        diff = (result_cf - x_in).abs().mean(dim=1, keepdim=True)
    return result_cf, diff


def compute_dff_loss_cf_steering(model, preprocess, img_real, img_gen, steps=80, lr=3e-2, tv=1e-3, downsample=16):
    xr = preprocess(img_real)
    xg = preprocess(img_gen)
    xr = F.interpolate(xr, size=(66, 200), mode="bilinear", align_corners=False)
    xg = F.interpolate(xg, size=(66, 200), mode="bilinear", align_corners=False)
    _, diff_r = _steering_counterfactual_robust(model, xr, steps=steps, lr=lr, tv=tv)
    _, diff_g = _steering_counterfactual_robust(model, xg, steps=steps, lr=lr, tv=tv)
    ds = max(4, int(downsample))
    diff_r_ds = F.adaptive_avg_pool2d(diff_r, (ds, ds))
    diff_g_ds = F.adaptive_avg_pool2d(diff_g, (ds, ds))
    return float(F.mse_loss(diff_r_ds, diff_g_ds).item())


def compute_dff_loss_cf_yolop(model, preprocess, img_real, img_gen, head="da", steps=80, lr=2e-2, tv=1e-3, downsample=16):
    xr = preprocess(img_real)
    xg = preprocess(img_gen)
    xr = F.interpolate(xr, size=(384, 640), mode="bilinear", align_corners=False)
    xg = F.interpolate(xg, size=(384, 640), mode="bilinear", align_corners=False)
    selector = yolop_select_da_head if head == "da" else yolop_select_ll_head
    _, diff_r = _yolop_counterfactual_robust(model, xr, selector, steps=steps, lr=lr, tv=tv)
    _, diff_g = _yolop_counterfactual_robust(model, xg, selector, steps=steps, lr=lr, tv=tv)
    ds = max(4, int(downsample))
    diff_r_ds = F.adaptive_avg_pool2d(diff_r, (ds, ds))
    diff_g_ds = F.adaptive_avg_pool2d(diff_g, (ds, ds))
    return float(F.mse_loss(diff_r_ds, diff_g_ds).item())


def main():
    if not (os.path.exists(REAL_IMG) and os.path.exists(GEN_IMG)):
        raise FileNotFoundError("Missing generated_figures/real_input.png or Baseline.png")
    img_real = Image.open(REAL_IMG).convert("RGB")
    img_gen = Image.open(GEN_IMG).convert("RGB")

    steering_model, steering_pre = load_steering_model()
    yolop_model, yolop_pre = load_yolop_model()

    rows = []
    k_list = [20, 40, 80]
    tv_list = [1e-3, 5e-3]

    for k in k_list:
        for tv in tv_list:
            loss_steer = compute_dff_loss_cf_steering(steering_model, steering_pre, img_real, img_gen, steps=k, tv=tv)
            rows.append({"sut": "steering", "k_cf": k, "tv": tv, "dff_loss": loss_steer, "real_img": REAL_IMG, "gen_img": GEN_IMG})
            loss_da = compute_dff_loss_cf_yolop(yolop_model, yolop_pre, img_real, img_gen, head="da", steps=k, tv=tv)
            rows.append({"sut": "da", "k_cf": k, "tv": tv, "dff_loss": loss_da, "real_img": REAL_IMG, "gen_img": GEN_IMG})
            loss_ll = compute_dff_loss_cf_yolop(yolop_model, yolop_pre, img_real, img_gen, head="ll", steps=k, tv=tv)
            rows.append({"sut": "ll", "k_cf": k, "tv": tv, "dff_loss": loss_ll, "real_img": REAL_IMG, "gen_img": GEN_IMG})

    out_csv = os.path.join(ROOT, "dff_sensitivity.csv")
    import pandas as pd
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print("dff_sensitivity_csv=", out_csv)


if __name__ == "__main__":
    main()
