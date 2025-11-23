#!/usr/bin/env python3

import argparse, csv, sys, random
from pathlib import Path

def read_rows(csv_path: Path):
    rows_pos, rows_neg = [], []
    with open(csv_path, "r") as f:
        r = csv.DictReader(f)
        if "path" not in r.fieldnames or "label" not in r.fieldnames:
            raise ValueError("CSV must contain 'path' and 'label' columns.")
        for row in r:
            lbl = int(row["label"])
            tup = (row["path"], lbl)
            if   lbl == 1: rows_pos.append(tup)
            elif lbl == 0: rows_neg.append(tup)
            else: raise ValueError(f"Label must be 0/1, got: {lbl}")
    return rows_pos, rows_neg

def write_csv(out_path: Path, rows):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path","label"])
        w.writerows(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp", required=True, help="Input CSV (must have path,label)")
    ap.add_argument("--out", required=True, help="Output CSV")
    ap.add_argument("--n",   type=int, required=True, help="Target number of rows")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    inp = Path(args.inp); outp = Path(args.out)
    rows_pos, rows_neg = read_rows(inp)

    n_pos = len(rows_pos); n_neg = len(rows_neg); n_tot = n_pos + n_neg
    if args.n > n_tot:
        print(f"Error: target n={args.n} exceeds available {n_tot}.", file=sys.stderr)
        sys.exit(1)
    if args.n <= 0:
        print("Error: n must be > 0.", file=sys.stderr); sys.exit(1)

    rng = random.Random(args.seed)

    ratio_pos = n_pos / n_tot if n_tot > 0 else 0.5
    tgt_pos = int(round(args.n * ratio_pos))
    tgt_neg = args.n - tgt_pos

    if tgt_pos > n_pos:
        deficit = tgt_pos - n_pos
        tgt_pos = n_pos
        tgt_neg = min(n_neg, tgt_neg + deficit)
    if tgt_neg > n_neg:
        deficit = tgt_neg - n_neg
        tgt_neg = n_neg
        tgt_pos = min(n_pos, tgt_pos + deficit)

    if tgt_pos + tgt_neg != args.n:
        remain = args.n - (tgt_pos + tgt_neg)
        if remain > 0:
            spare_pos = n_pos - tgt_pos
            add_pos = min(spare_pos, remain)
            tgt_pos += add_pos
            tgt_neg += (remain - add_pos)
        elif remain < 0:
            drop = -remain
            drop_pos = min(tgt_pos, drop)
            tgt_pos -= drop_pos
            tgt_neg -= (drop - drop_pos)

    assert 0 <= tgt_pos <= n_pos and 0 <= tgt_neg <= n_neg and (tgt_pos + tgt_neg) == args.n

    samp_pos = rng.sample(rows_pos, tgt_pos) if tgt_pos > 0 else []
    samp_neg = rng.sample(rows_neg, tgt_neg) if tgt_neg > 0 else []

    merged = samp_pos + samp_neg
    rng.shuffle(merged)

    write_csv(outp, merged)

    print(f"input : pos={n_pos} neg={n_neg} total={n_tot}")
    print(f"target: n={args.n} -> pos={tgt_pos} neg={tgt_neg}")
    print(f"wrote : {outp}")

if __name__ == "__main__":
    main()
