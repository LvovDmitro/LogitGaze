# LogitGaze

LogitGaze is a transformer-based model for scanpath / fixation prediction that integrates:

- **Transformer-based scanpath model** — predicts full fixation sequences conditioned on the image and search target.
- **Logit Lens features** — semantic vectors extracted from a vision–language model (LLaVA) for each image patch.

This directory is the standalone, cleaned codebase for the **LogitGaze** paper:

- *“LogitGaze: Predicting Human Attention Using Semantic Information from Vision-Language Models”* (Lvov, Pershin), ICLR 2025 – OpenReview: `https://openreview.net/forum?id=Om4j6IVb42`
- *“LogitGaze-Med: Semantic-Guided Scanpath Prediction in Medical Imaging”* (Lvov, Pershin et al.), OpenReview: `https://openreview.net/forum?id=lTAtQW1Ufi`

## 1. Project Layout

```
LogitGaze/
├── models/               # LogitGaze model architecture
├── utils/                # Datasets, feature extraction, CLI args
├── metrics/              # Evaluation metrics
├── training_scripts/     # train.py, test.py
├── logit_lens/           # Logit Lens generation (LLaVA-based)
├── data/                 # (you create this) inputs for training
└── configs/              # optional configs (empty by default)
```

## 2. What you need to add (not in repo)

The repo **does not contain** any data or weights. You must provide:

- `data/dataset_images/` – COCO-Search18 images (TP/TA folders etc.)
- `dataset/` – COCO-Search18 metadata and precomputed features:
  - `dataset/coco_search18_fixations_TP_train.json`
  - `dataset/coco_search18_fixations_TP_validation.json`
  - `dataset/coco_search18_fixations_TP_test.json`
  - `dataset/embeddings.npy`
  - `dataset/image_features/*.pth` (ResNet features per image)
- `data/logit_lens_TP/` – generated Logit Lens vectors:
  - `data/logit_lens_TP/logits/*.json`
  - `data/logit_lens_TP/semantics/*.npy`
- `saved_models/` (optional) – your trained checkpoints if you want to share them

If you copy from your existing environment, you will have:

```bash
cp -r /path/to/original/dataset      /repo/LogitGaze/dataset
cp -r /path/to/original/data         /repo/LogitGaze/data
cp -r /path/to/original/saved_models /repo/LogitGaze/saved_models   # optional
```

## 3. Generate Logit Lens vectors

From repo root:

```bash
cd /repo/LogitGaze

python logit_lens/scripts/logit_lens/logit_lens_gaze.py \
  --image_folder "./data/dataset_images/images_TP" \
  --save_folder "./data/logit_lens_TP" \
  --device "cuda:0"
```

Or use the notebook:

```bash
jupyter notebook logit_lens/notebooks/generate_and_visualize_logit_lens.ipynb
```

## 4. Train with Logit Lens

```bash
cd /repo/LogitGaze

python training_scripts/train.py \
  --use_logit_lens \
  --dataset_dir "./dataset" \
  --img_ftrs_dir "./dataset/image_features" \
  --logit_lens_dir "./data/logit_lens_TP"
```

Key defaults (can be overridden):
- `logit_lens_top_k=5`
- `logit_lens_dim=4096`
- `batch_size=32`
- `epochs=200`

## 5. Test with the latest trained model

Assuming you trained and have a checkpoint at:

```text
saved_models/trained/train_13-01-2026-09-56-20/logitgaze_6E_6D_16_512d_27.pkg
```

Run:

```bash
cd /repo/LogitGaze

python training_scripts/test.py \
  --use_logit_lens \
  --trained_model "./saved_models/trained/train_13-01-2026-09-56-20/logitgaze_6E_6D_16_512d_27.pkg" \
  --dataset_dir "./dataset" \
  --img_ftrs_dir "./dataset/image_features" \
  --logit_lens_dir "./data/logit_lens_TP" \
  --logit_lens_top_k 5
```

## 6. Notes for GitHub

- You can safely commit **all files in `LogitGaze/`** except:
  - `data/` (large and dataset-dependent)
  - `dataset/` (COCO-Search18, license-restricted)
  - `saved_models/` (large checkpoints)
- Add a `.gitignore` with at least:

```gitignore
data/
dataset/
saved_models/
logs/
*.pth
*.npy
*.npz
*.pkg
```

This makes the repo clean, light, and ready to share as the official **LogitGaze** codebase accompanying the papers listed above.

