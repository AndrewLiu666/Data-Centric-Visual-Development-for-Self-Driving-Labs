#!/usr/bin/env python3

import argparse
import csv
import math
import random
from pathlib import Path
from typing import List, Tuple

from sklearn.model_selection import train_test_split

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def list_images(folder: Path) -> List[str]:
    out = []
    for p in sorted(folder.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in EXTS:
            continue
        name = p.name
        if name.startswith("._") or name.startswith("."):
            continue
        if "__MACOSX" in p.parts:
            continue
        try:
            if p.stat().st_size < 4096:
                continue
        except Exception:
            continue
        out.append(str(p))
    return out

def write_csv(path: Path, rows: List[Tuple[str, int]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])
        w.writerows(rows)

def normalize_ratios(r_train: float, r_val: float, r_test: float) -> Tuple[float, float, float]:
    r_train = max(0.0, float(r_train))
    r_val   = max(0.0, float(r_val))
    r_test  = max(0.0, float(r_test))
    s = r_train + r_val + r_test
    if s == 0.0:
        raise ValueError("All ratios are zero. Provide at least one positive ratio.")
    return r_train/s, r_val/s, r_test/s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", required=True, type=str, help="Folder of has_bubble images (label=1)")
    ap.add_argument("--neg", required=True, type=str, help="Folder of no_bubble images (label=0)")
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.15, 0.15],
                    help="Ratios for train val test (can be zero), e.g. 0.7 0.15 0.15")
    ap.add_argument("--out", type=str, default="splits", help="Output folder for CSVs")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    random.seed(args.seed)

    pos_dir = Path(args.pos)
    neg_dir = Path(args.neg)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    assert pos_dir.exists() and pos_dir.is_dir(), f"pos folder not found: {pos_dir}"
    assert neg_dir.exists() and neg_dir.is_dir(), f"neg folder not found: {neg_dir}"

    pos = list_images(pos_dir)
    neg = list_images(neg_dir)
    assert len(pos) > 0 and len(neg) > 0, "No images found in given folders."

    X = pos + neg
    y = [1]*len(pos) + [0]*len(neg)

    r_tr, r_va, r_te = normalize_ratios(*args.ratios)

    test_size_total = 1.0 - r_tr
    if test_size_total < 1e-8:
        X_tr, y_tr = X, y
        X_rest, y_rest = [], []
    else:
        X_tr, X_rest, y_tr, y_rest = train_test_split(
            X, y, test_size=test_size_total, stratify=y, random_state=args.seed
        )

    X_va, X_te, y_va, y_te = [], [], [], []
    if test_size_total > 1e-8:
        if r_va < 1e-8 and r_te > 0:
            X_te, y_te = X_rest, y_rest
        elif r_te < 1e-8 and r_va > 0:
            X_va, y_va = X_rest, y_rest
        elif r_va < 1e-8 and r_te < 1e-8:
            pass
        else:
            frac_val = r_va / (r_va + r_te)
            X_va, X_te, y_va, y_te = train_test_split(
                X_rest, y_rest, test_size=(1.0 - frac_val),
                stratify=y_rest, random_state=args.seed
            )

    write_csv(out_dir / "train.csv", list(zip(X_tr, y_tr)))
    write_csv(out_dir / "val.csv",   list(zip(X_va, y_va)))
    write_csv(out_dir / "test.csv",  list(zip(X_te, y_te)))

    print(f"Train/Val/Test = {len(X_tr)}/{len(X_va)}/{len(X_te)}")
    def count_lbl(lbls, v): return sum(1 for t in lbls if t == v)
    print(f"  Train: pos={count_lbl(y_tr,1)} neg={count_lbl(y_tr,0)}")
    print(f"  Val  : pos={count_lbl(y_va,1)} neg={count_lbl(y_va,0)}")
    print(f"  Test : pos={count_lbl(y_te,1)} neg={count_lbl(y_te,0)}")

if __name__ == "__main__":
    main()
