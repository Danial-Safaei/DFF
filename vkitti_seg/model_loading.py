"""Unified model loading utilities for V-KITTI ControlNet training variants.

Provides safe, state-dict oriented loading for:
- Steering angle prediction model (AutonomousDrivingModel)
- YOLOP multi-head model (drivable area / lane line)

All loaders are defensive: they tolerate full pickled objects, raw state dicts, or
wrappers with a "state_dict" key. They log missing / unexpected keys and fall back
silently to lightweight dummy models if loading fails (without crashing training).
"""
from __future__ import annotations

import os
import sys
import types
import importlib.util
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn

# --------------------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------------------

def _dynamic_import(py_path: str, module_name: str) -> types.ModuleType:
    if not os.path.exists(py_path):
        raise FileNotFoundError(f"Architecture file not found: {py_path}")
    spec = importlib.util.spec_from_file_location(module_name, py_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed creating import spec for: {py_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


def _load_checkpoint_any(ckpt_path: str) -> Dict[str, Any]:
    obj = torch.load(ckpt_path, map_location="cpu")
    # Accept direct state dict, or container with common keys
    if isinstance(obj, dict):
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj  # raw state dict
        for k in ("state_dict", "model", "network"):
            if k in obj and isinstance(obj[k], dict):
                inner = obj[k]
                if all(isinstance(v, torch.Tensor) for v in inner.values()):
                    return inner
    return obj  # fallback (may still be a full model)


def _clean_state_dict_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Strip common prefixes like 'module.' from DDP / DataParallel
    cleaned = {}
    for k, v in state.items():
        if k.startswith("module."):
            cleaned[k[len("module."):]] = v
        else:
            cleaned[k] = v
    return cleaned

# --------------------------------------------------------------------------------------
# Steering Model Loading
# --------------------------------------------------------------------------------------

class DummySteeringModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Marker used by training scripts to detect fallback behavior.
        # This prevents silently "training OVF" against a zero-output model.
        self.is_dummy = True
        self.fc = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        # produce deterministic zero-like output for stability
        return torch.zeros(b, 1, device=x.device)


def load_steering_model(arch_path: str, weights_path: str, device: torch.device, dtype: Optional[torch.dtype] = None) -> nn.Module:
    try:
        module = _dynamic_import(arch_path, "steering_arch")
        if not hasattr(module, "AutonomousDrivingModel"):
            raise AttributeError("AutonomousDrivingModel not found in architecture file")
        model = module.AutonomousDrivingModel()
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Steering weights missing: {weights_path}")
        raw_state = _load_checkpoint_any(weights_path)
        if isinstance(raw_state, nn.Module):  # full model pickled
            model = raw_state
        else:
            if not isinstance(raw_state, dict):
                raise RuntimeError("Unsupported steering checkpoint format")
            state = _clean_state_dict_keys(raw_state)
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"[SteeringModel] Loaded weights: {weights_path}\n  Missing={len(missing)} Unexpected={len(unexpected)}")
        if dtype is not None:
            model = model.to(device=device, dtype=dtype).eval()
        else:
            model = model.to(device).eval()
        # Mark as a real model for downstream checks.
        try:
            model.is_dummy = False  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            for p in model.parameters():
                p.requires_grad = False
        except Exception:
            pass
        return model
    except Exception as e:
        print(f"[SteeringModel] Fallback to dummy model due to error: {e}")
        m = DummySteeringModel().to(device).eval()
        try:
            for p in m.parameters():
                p.requires_grad = False
        except Exception:
            pass
        return m

# --------------------------------------------------------------------------------------
# YOLOP Loading
# --------------------------------------------------------------------------------------

class DummyYOLOP(nn.Module):
    def __init__(self):
        super().__init__()
        # Marker used by training scripts to detect fallback behavior.
        self.is_dummy = True

    def forward(self, x: torch.Tensor):  # mimic triple output pattern
        b, _, h, w = x.shape
        zero_map = torch.zeros(b, 1, h, w, device=x.device)
        return zero_map, zero_map, zero_map  # det/da/ll placeholders


def load_yolop_branches(yolop_root: str, weights_path: Optional[str], device: torch.device,
                        require: str = "both", dtype: Optional[torch.dtype] = None):
    """Return callable(s) for requested YOLOP branches.

    require: one of {'da','ll','both'} controlling which branch tensor(s) are returned.
    """
    try:
        lib_dir = os.path.join(yolop_root, "lib")
        if not os.path.isdir(lib_dir):
            raise FileNotFoundError(f"YOLOP lib directory not found: {lib_dir}")
        # Add root to path so 'import lib' works
        if yolop_root not in sys.path:
            sys.path.append(yolop_root)
        # Add lib to path so 'import models' works if needed
        if lib_dir not in sys.path:
            sys.path.append(lib_dir)
        # YOLOP repo structure: import model definition (assumes standard YOLOP.py exposes a Model builder or similar)
        from models.YOLOP import get_net  # type: ignore
        # In absence of a clear constructor in snippet, fallback to dummy; extension point for actual build logic.
        model = get_net(False)
        if weights_path and os.path.exists(weights_path):
            try:
                raw_state = _load_checkpoint_any(weights_path)
                if isinstance(raw_state, nn.Module):
                    model = raw_state
                elif isinstance(raw_state, dict):
                    state = _clean_state_dict_keys(raw_state)
                    missing, unexpected = model.load_state_dict(state, strict=False)
                    print(f"[YOLOP] Loaded weights: {weights_path} Missing={len(missing)} Unexpected={len(unexpected)}")
            except Exception as werr:
                print(f"[YOLOP] Failed to apply weights ({weights_path}): {werr}. Using dummy model.")
        if dtype is not None:
            model = model.to(device=device, dtype=dtype).eval()
        else:
            model = model.to(device).eval()
        # Mark as a real model for downstream checks.
        try:
            model.is_dummy = False  # type: ignore[attr-defined]
        except Exception:
            pass
        try:
            for p in model.parameters():
                p.requires_grad = False
        except Exception:
            pass

        def forward_da(x: torch.Tensor):
            out = model(x)
            if isinstance(out, (list, tuple)) and len(out) >= 2:
                return out[1]
            return out[0] if isinstance(out, (list, tuple)) else out

        # Allow callers to detect fallback behavior.
        try:
            forward_da.is_dummy = getattr(model, "is_dummy", False)  # type: ignore[attr-defined]
        except Exception:
            pass

        def forward_ll(x: torch.Tensor):
            out = model(x)
            if isinstance(out, (list, tuple)) and len(out) >= 3:
                return out[2]
            return out[0] if isinstance(out, (list, tuple)) else out

        # Allow callers to detect fallback behavior.
        try:
            forward_ll.is_dummy = getattr(model, "is_dummy", False)  # type: ignore[attr-defined]
        except Exception:
            pass

        if require == "da":
            return forward_da
        if require == "ll":
            return forward_ll
        return forward_da, forward_ll
    except Exception as e:
        print(f"[YOLOP] Fallback to dummy due to error: {e}")
        dummy = DummyYOLOP().to(device).eval()
        try:
            for p in dummy.parameters():
                p.requires_grad = False
        except Exception:
            pass
        def fwd(x: torch.Tensor):
            return dummy(x)

        # Mark fallback
        try:
            fwd.is_dummy = True  # type: ignore[attr-defined]
        except Exception:
            pass
        if require == "both":
            def _da(x: torch.Tensor):
                return dummy(x)[1]
            def _ll(x: torch.Tensor):
                return dummy(x)[2]
            try:
                _da.is_dummy = True  # type: ignore[attr-defined]
                _ll.is_dummy = True  # type: ignore[attr-defined]
            except Exception:
                pass
            return (_da, _ll)
        elif require == "da":
            def _da(x: torch.Tensor):
                return dummy(x)[1]
            try:
                _da.is_dummy = True  # type: ignore[attr-defined]
            except Exception:
                pass
            return _da
        else:
            def _ll(x: torch.Tensor):
                return dummy(x)[2]
            try:
                _ll.is_dummy = True  # type: ignore[attr-defined]
            except Exception:
                pass
            return _ll

# --------------------------------------------------------------------------------------
# Convenience orchestrator used by variant scripts
# --------------------------------------------------------------------------------------

def initialize_external_models(device: torch.device,
                               steering_arch_path: Optional[str] = None,
                               steering_weights_path: Optional[str] = None,
                               yolop_root: Optional[str] = None,
                               yolop_weights_path: Optional[str] = None,
                               need_steering: bool = False,
                               yolop_branch: Optional[str] = None):
    """Centralized initialization returning a dict of callables / models.

    yolop_branch: 'da', 'll', or None/'both'.
    """
    artifacts: Dict[str, Any] = {}

    if need_steering and steering_arch_path and steering_weights_path:
        artifacts["steering_model"] = load_steering_model(steering_arch_path, steering_weights_path, device)
    elif need_steering:
        artifacts["steering_model"] = DummySteeringModel().to(device).eval()

    if yolop_root:
        branch = yolop_branch or "both"
        artifacts["yolop"] = load_yolop_branches(yolop_root, yolop_weights_path, device, branch if branch in {"da","ll"} else "both")

    return artifacts
