#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import timm
from dataset import build_loader

CFG = {
    "img_size_hw": (600, 1500),
    "batch_size": 2,
    "epochs": 300,
    "patience": 3,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "model_name": "tf_efficientnetv2_l.in21k_ft_in1k",
    "num_workers": 4,
    "tta": False
}

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def build_model():
    return timm.create_model(CFG["model_name"], pretrained=True, num_classes=1)

@torch.no_grad()
def collect_probs_labels(model, loader, device, tta=False):
    model.eval()
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.cpu().numpy().ravel()
        with autocast('cuda'):
            logits = model(imgs).squeeze(1)
            probs = torch.sigmoid(logits).float().cpu().numpy()
            if tta:
                imgs_f = torch.flip(imgs, dims=[3])
                logits_f = model(imgs_f).squeeze(1)
                probs_f = torch.sigmoid(logits_f).float().cpu().numpy()
                probs = (probs + probs_f) / 2.0
        all_probs.append(probs)
        all_labels.append(labels)
    return np.concatenate(all_probs), np.concatenate(all_labels)

def eval_at_threshold(probs, labels, thr):
    preds = (probs >= thr).astype(int)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tp = int(((preds == 1) & (labels == 1)).sum())
    fpr = fp / max(1, (fp + tn))
    return {
        "thr": float(thr),
        "acc": float(acc),
        "prec": float(p),
        "rec": float(r),
        "f1": float(f1),
        "fpr": float(fpr),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn
    }

def build_threshold_grid(probs=None):
    hi = np.array([0.95, 0.98, 0.99, 0.995, 0.999, 1.0, 1.0001])
    lo = np.concatenate([
        np.array([0.0]),
        np.geomspace(1e-4, 5e-3, 20),
        np.linspace(0.005, 0.05, 20),
        np.linspace(0.05, 0.95, 38)
    ])
    if probs is not None:
        return np.unique(np.r_[probs, lo, hi])
    return np.unique(np.r_[lo, hi])

def pick_threshold_by_f1(probs, labels):
    grid = build_threshold_grid(probs)
    report = [eval_at_threshold(probs, labels, thr) for thr in grid]
    best = max(report, key=lambda r: r["f1"])
    return {"kind": "f1", "metrics": best, "grid_report": report}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=str, required=True,
                        help="Folder containing train.csv / val.csv")
    parser.add_argument("--out", type=str, required=True,
                        help="Output folder for checkpoints & logs")
    parser.add_argument("--tta", action="store_true",
                        help="Enable horizontal-flip TTA on validation")
    args = parser.parse_args()

    if args.tta:
        CFG["tta"] = True

    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    SPLIT_DIR = Path(args.splits)
    OUT_DIR = Path(args.out)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_loader = build_loader(SPLIT_DIR / "train.csv", "train",
                                CFG["batch_size"], CFG["img_size_hw"], CFG["num_workers"])
    val_loader = build_loader(SPLIT_DIR / "val.csv", "val",
                              CFG["batch_size"], CFG["img_size_hw"], CFG["num_workers"])

    model = build_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG["epochs"])
    scaler = GradScaler('cuda')

    best_f1 = -1.0
    best_pack = None
    epochs_no_improve = 0

    for epoch in range(1, CFG["epochs"] + 1):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{CFG['epochs']}")
        for imgs, labels in pbar:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).view(-1)

            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                logits = model(imgs).squeeze(1)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        train_loss = running_loss / max(1, len(train_loader.dataset))

        probs, labels = collect_probs_labels(model, val_loader, device, tta=CFG["tta"])
        sel = pick_threshold_by_f1(probs, labels)
        m = sel["metrics"]
        ref_05 = eval_at_threshold(probs, labels, 0.5)

        print(
            f"[Val] epoch={epoch} loss={train_loss:.4f} "
            f"thr*={m['thr']:.4f} f1={m['f1']:.4f} acc={m['acc']:.4f} "
            f"prec={m['prec']:.4f} rec={m['rec']:.4f} fpr={m['fpr']:.4f} "
            f"tp={m['tp']} fp={m['fp']} tn={m['tn']} fn={m['fn']} "
            f"| @0.50 f1={ref_05['f1']:.4f} rec={ref_05['rec']:.4f} fpr={ref_05['fpr']:.4f}"
        )

        improved = m["f1"] > best_f1
        if improved:
            best_f1 = m["f1"]
            epochs_no_improve = 0
            best_pack = {
                "model": model.state_dict(),
                "cfg": CFG,
                "best_thr": float(m["thr"]),
                "best_kind": "f1",
                "best_metrics": m,
                "epoch": epoch,
            }
            torch.save(best_pack, OUT_DIR / "best.pt")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= CFG["patience"]:
            print(f"Early stopping at epoch {epoch}. Best F1 = {best_f1:.4f} "
                  f"(thr={best_pack.get('best_thr','?')})")
            break

    summary = {
        "best_f1": float(best_f1),
        "best_kind": best_pack["best_kind"] if best_pack else None,
        "best_thr": float(best_pack["best_thr"]) if best_pack else None,
        "best_metrics": best_pack["best_metrics"] if best_pack else None,
        "tta": bool(CFG["tta"]),
        "epochs_trained": int(best_pack["epoch"]) if best_pack else 0
    }
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
