import argparse, csv, json
from pathlib import Path
import numpy as np
import torch, timm
from PIL import Image, ImageOps
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def letterbox_rect(img, size_hw):
    H, W = size_hw
    w, h = img.size
    RESIZE_BILINEAR = Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
    s = min(H / h, W / w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    img = img.resize((nw, nh), RESIZE_BILINEAR)
    pad_w, pad_h = W - nw, H - nh
    pad = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
    return ImageOps.expand(img, border=pad, fill=0)

def load_model(ckpt_path: Path):
    ck = torch.load(ckpt_path, map_location="cpu")
    cfg = ck["cfg"]
    thr = float(ck.get("best_thr", 0.5))
    model = timm.create_model(cfg["model_name"], pretrained=False, num_classes=1)
    model.load_state_dict(ck["model"], strict=True)
    model.eval()
    return model, (cfg["img_size_hw"][0], cfg["img_size_hw"][1]), thr, cfg

def read_test_csv(p):
    rows = []
    with open(p, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({"path": row["path"], "label": int(row["label"]), "group": row.get("group", None)})
    return rows

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", required=True, help="Folder containing test.csv")
    ap.add_argument("--ckpt", required=True, help="Path to best.pt")
    ap.add_argument("--out",  required=True, help="Output folder for metrics files")
    ap.add_argument("--tta", action="store_true", help="Enable horizontal-flip TTA")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, size_hw, thr, cfg = load_model(Path(args.ckpt))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    tx = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    test_rows = read_test_csv(Path(args.splits) / "test.csv")
    y_true, y_prob, y_pred = [], [], []

    for row in test_rows:
        img = Image.open(row["path"]).convert("RGB")
        img = letterbox_rect(img, size_hw)
        x = tx(img).unsqueeze(0).to(device)

        p0 = torch.sigmoid(model(x).squeeze(1)).item()
        if args.tta:
            x_flip = torch.flip(x, dims=[3])
            p1 = torch.sigmoid(model(x_flip).squeeze(1)).item()
            prob = 0.5 * (p0 + p1)
        else:
            prob = p0

        y = row["label"]
        z = int(prob >= thr)
        y_true.append(y)
        y_prob.append(prob)
        y_pred.append(z)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    fpr = fp / max(1, (fp + tn))

    out = {
        "threshold": thr,
        "tta": bool(args.tta),
        "frame_level": {
            "acc": float(acc),
            "prec": float(prec),
            "rec": float(rec),
            "f1": float(f1),
            "fpr": float(fpr),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn
        },
        "n_frames": len(test_rows)
    }
    print(json.dumps(out, indent=2))

    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(out, f, indent=2)

    with open(out_dir / "test_frame_probs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "prob", "pred"])
        for row, p, z in zip(test_rows, y_prob, y_pred):
            w.writerow([row["path"], row["label"], p, int(z)])

if __name__ == "__main__":
    main()
