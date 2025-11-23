# Data-Centric Visual Development for Self-Driving Labs

# Installation
- **GPU Memory > 12GB**
- Python = 3.9
- PyTorch = 2.8.0


```bash
git clone https://github.com/AndrewLiu666/Data-Centric-Visual-Development-for-Self-Driving-Labs.git
conda create -n vision python=3.9
cd Data-Centric-Visual-Development-for-Self-Driving-Labs
conda activate vision
pip install -r requirements.txt
```

Dataset can be download from [dataset](https://drive.google.com/file/d/1PJ30lIOCOF9ies4koOLjkEx-ocbPL7i0/view?usp=share_link).

# Pre-training
Use prepare_split.py to
- Recursively scans two folders:
  - --pos: images for label 1 (has_bubble)
  - --neg: images for label 0 (no_bubble)
- Filters valid image files and ignores tiny/hidden/macOS sidecar files.
- Stratified splits the combined dataset into train/val/test according to user-specified ratios.
- Writes train.csv, val.csv, and test.csv into an output directory.
```bash
python src/prepare_split.py \
  --pos data/roi/has_bubble \
  --neg data/roi/no_bubble \
  --ratios 0.7 0.15 0.15 \
  --out splits/real_main \
  --seed 42
```

Use merge_train_csv.py to
- Merges two train CSVs (e.g., real + synthetic) into a single CSV.
- Keeps only path and label from each input CSV.
```bash
python src/merge_train_csv.py \
  --real  splits/real_main/train.csv \
  --synth splits/synth_main/train.csv \
  --out   splits/mixed/train.csv
```

Use subsample_csv_stratified.py to
- Takes an existing CSV with path,label.
- Stratified subsamples exactly N rows while approximately preserving the original class ratio (0 vs 1).
- Handles edge cases where one class does not have enough samples and adjusts the other class accordingly.
```bash
python src/subsample_csv_stratified.py \
  --in  splits/real_main/train.csv \
  --out splits/real_main/train_2240.csv \
  --n   2240 \
  --seed 42
```

# Train
Use train.py to
- Trains an EfficientNetV2-L model (tf_efficientnetv2_l.in21k_ft_in1k via timm) for binary classification.
- Saves best.pt and writes a summary.json with the best metrics.
```bash
python src/train.py \
  --splits splits/real_main \
  --out runs/effv2l_600x1500 \
  --tta
```

# Evaluate
Use eval_test.py to 
- Evaluate the model with the best.pt from training
```bash
python src/eval_test.py \
  --splits splits/real_main \
  --ckpt runs/effv2l_600x1500/best.pt \
  --out  runs/effv2l_600x1500 \
  --tta
```