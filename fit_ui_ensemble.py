# -*- coding: utf-8 -*-
"""Fit guarded val bias + val-accuracy weights for the UI ensemble."""
import os, json
import numpy as np
import torch
import sys
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "DL_models"))
from DL_models.multimodal_dataloader import get_upgraded_dataloaders
from DL_models.dual_branch_network import get_upgraded_model, LEGACY_GEOM
import ui_ensemble as UE

def _load(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    bb = ck.get("backbone", "resnet")
    io = ck.get("image_only", None)
    st = ck["model_state_dict"] if "model_state_dict" in ck else ck
    if io is None:
        io = not any(k.startswith("blendshape_projector.") for k in st)
    leg = ("blendshape_projector.fc.0.weight" in st)
    kw = dict(LEGACY_GEOM) if leg else {}
    if "fusion_head.1.weight" in st and int(st["fusion_head.1.weight"].shape[0]) == 256 and not leg:
        kw = dict(LEGACY_GEOM)
    m = get_upgraded_model(backbone_type=bb, num_classes=3, image_only=bool(io), **kw)
    try:
        m.load_state_dict(st)
    except RuntimeError:
        from DL_models.dual_branch_network import StudentEngagementDualBranchModel as M
        hid = int(st["blendshape_projector.fc.0.weight"].shape[0])
        outd = int(st["blendshape_projector.fc.5.weight"].shape[0])
        fus = int(st["fusion_head.1.weight"].shape[0])
        m = M(backbone_type=bb, num_classes=3, fusion_hidden_dim=fus,
              dropout_rate=0.25, pretrained=True, image_only=False,
              geom_hidden_dim=hid, geom_out_dim=outd)
        md = m.state_dict()
        mp = {"blendshape_projector.fc.0.weight": "blendshape_projector.in_proj.0.weight",
              "blendshape_projector.fc.0.bias": "blendshape_projector.in_proj.0.bias",
              "blendshape_projector.fc.1.weight": "blendshape_projector.res_block.0.weight",
              "blendshape_projector.fc.1.bias": "blendshape_projector.res_block.0.bias",
              "blendshape_projector.fc.4.weight": "blendshape_projector.out_proj.0.weight",
              "blendshape_projector.fc.4.bias": "blendshape_projector.out_proj.0.bias",
              "blendshape_projector.fc.5.weight": "blendshape_projector.out_proj.0.weight",
              "blendshape_projector.fc.5.bias": "blendshape_projector.out_proj.0.bias"}
        for a, b in mp.items():
            if a in st and b in md and md[b].shape == st[a].shape:
                md[b] = st[a]
        for k, v in st.items():
            if k in md and md[k].shape == v.shape and not k.startswith("blendshape_projector.fc."):
                md[k] = v
        m.load_state_dict(md, strict=False)
    m.to(device).eval()
    return m, bool(io)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader, _, _ = get_upgraded_dataloaders(batch_size=64)
    paths = UE.discover_checkpoints()
    print(f"checkpoints ({len(paths)}):")
    for p in paths:
        print("  -", os.path.basename(p))
    models = [_load(p, device) for p in paths]
    Ls, Y = [], []
    with torch.no_grad():
        for imgs, bs, labels, _ in val_loader:
            imgs, bs = imgs.to(device), bs.to(device)
            Y.append(labels.numpy())
            b = []
            for m, io in models:
                lg = m(imgs) if io else m(imgs, bs)
                b.append(torch.softmax(lg, dim=1).cpu().numpy())
            Ls.append(np.stack(b, axis=1))
    L = np.concatenate(Ls, axis=0)
    Y = np.concatenate(Y, axis=0)
    vacc = np.array([(L[:, i, :].argmax(1) == Y).mean() for i in range(L.shape[1])])
    print("member val acc:", [round(float(x) * 100, 2) for x in vacc])
    w = vacc / vacc.sum()
    ens_log = np.log((L * w[None, :, None]).sum(1) + 1e-12)
    rng = np.random.RandomState(42)
    idx = np.arange(len(Y))
    rng.shuffle(idx)
    halves = np.array_split(idx, 2)
    base_h = [float((ens_log[h].argmax(1) == Y[h]).mean()) for h in halves]
    base_full = float((ens_log.argmax(1) == Y).mean())
    cand = np.linspace(-0.6, 0.6, 25)
    best_gain, best_b = 0.0, np.zeros(2)
    for b0 in cand:
        for b1 in cand:
            adj = ens_log.copy()
            adj[:, 0] += b0
            adj[:, 1] += b1
            p = adj.argmax(1)
            gains = [float((p[h] == Y[h]).mean()) - base_h[hi] for hi, h in enumerate(halves)]
            mg = min(gains)
            full = float((p == Y).mean()) - base_full
            if mg > 1e-9 and full > 1e-9 and mg + full > best_gain + 1e-12:
                best_gain = mg + full
                best_b = np.array([b0, b1])
    print(f"base val={base_full*100:.2f}% halves={[round(x*100,2) for x in base_h]}")
    print(f"bias={best_b.tolist()} gain={best_gain*100:.3f}%")
    bias = np.array([float(best_b[0]), float(best_b[1]), 0.0])
    cal = ens_log.copy()
    cal[:, 0] += bias[0]
    cal[:, 1] += bias[1]
    cal_acc = float((cal.argmax(1) == Y).mean())
    print(f"calibrated val={cal_acc*100:.2f}%")
    cfg = {"checkpoints": paths, "weights": [float(x) for x in w],
           "member_val_acc": [float(x) for x in vacc],
           "bias": [float(x) for x in bias],
           "val_raw_acc": float(base_full), "val_cal_acc": float(cal_acc), "seed": 42}
    with open(os.path.join(BASE, "DL_models", "hyper-1", "ensemble_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print("wrote ensemble_config.json")


if __name__ == "__main__":
    main()

