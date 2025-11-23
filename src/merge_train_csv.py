#!/usr/bin/env python3

import argparse, csv
from pathlib import Path

def read_rows(csv_path: Path):
    rows = []
    with open(csv_path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append((row["path"], int(row["label"])))
    return rows

def write_csv(out_path: Path, rows):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label"])
        w.writerows(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", required=True, help="Path to real train.csv")
    ap.add_argument("--synth", required=True, help="Path to synthetic train.csv")
    ap.add_argument("--out", required=True, help="Output CSV path")
    args = ap.parse_args()

    real_rows = read_rows(Path(args.real))
    synth_rows = read_rows(Path(args.synth))
    merged = real_rows + synth_rows

    write_csv(Path(args.out), merged)

    print(f"merged -> {args.out}")
    print(f"  real : {len(real_rows)}")
    print(f"  synth: {len(synth_rows)}")
    print(f"  total: {len(merged)}")

if __name__ == "__main__":
    main()
