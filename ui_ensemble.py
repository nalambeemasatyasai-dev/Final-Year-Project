# -*- coding: utf-8 -*-
"""Live-UI calibrated ensemble (anti-overfit): 8 checkpoints x 6-view TTA."""
import os
import glob
import json
import numpy as np
import torch
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))


def discover_checkpoints():
    hyper = os.path.join(BASE, "DL_models", "hyper-1")
    paths = sorted(glob.glob(os.path.join(hyper, "*.pth")))
    keep = [p for p in paths if os.path.basename(p).startswith(
        ("best_model_upgraded_", "best_model_image_only_"))
        and "backup" not in os.path.basename(p).lower()]
    return keep or paths


def get_weights_bias():
    cfg_path = os.path.join(BASE, "DL_models", "hyper-1", "ensemble_config.json")
    n = len(discover_checkpoints())
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        w = np.asarray(cfg.get("weights", []), dtype=np.float64)
        if w.shape[0] != n or n == 0:
            w = np.ones(max(n, 1)) / max(n, 1)
        b = np.asarray(cfg.get("bias", [0.0, 0.0, 0.0]), dtype=np.float64)
        if b.shape[0] != 3:
            b = np.zeros(3)
        return w / w.sum(), b
    except Exception:
        w = np.ones(max(n, 1)) / max(n, 1)
        return w, np.zeros(3)


def tta_tensors(cropped_pil):
    cropped_pil = cropped_pil.convert("RGB")
    w, h = cropped_pil.size
    cw, ch = max(1, int(w * 0.88)), max(1, int(h * 0.88))
    l, t = (w - cw) // 2, (h - ch) // 2
    c1 = cropped_pil.crop((l, t, l + cw, t + ch))
    cw2, ch2 = max(1, int(w * 0.95)), max(1, int(h * 0.95))
    l2, t2 = (w - cw2) // 2, (h - ch2) // 2
    c2 = cropped_pil.crop((l2, t2, l2 + cw2, t2 + ch2))
    try:
        from torchvision import transforms as T
        norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        views = []
        for base in (cropped_pil, c1, c2):
            views.append((norm(T.ToTensor()(base)), False))
            try:
                m = base.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            except AttributeError:
                m = base.transpose(Image.FLIP_LEFT_RIGHT)
            views.append((norm(T.ToTensor()(m)), True))
        return views
    except Exception:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        def _t(p):
            a = np.array(p, dtype=np.float32) / 255.0
            return torch.from_numpy(((a - mean) / std).transpose(2, 0, 1))
        views = []
        for base in (cropped_pil, c1, c2):
            views.append((_t(base), False))
            try:
                m = base.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            except AttributeError:
                m = base.transpose(Image.FLIP_LEFT_RIGHT)
            views.append((_t(m), True))
        return views
