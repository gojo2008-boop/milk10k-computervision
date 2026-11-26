# MILK10K Skin Lesion Classification

Educational, non-commercial 11-class dermoscopic image-classification coursework project.

## Selected model
The selected experiment is the moderate-sampling ResNet18 checkpoint:
`checkpoints/resnet18_moderate_best.pth`.

Recorded results:
- Validation argmax macro-F1: 0.3812
- Local held-out test argmax macro-F1: 0.3702
- Official leaderboard score: 0.344 (separate protocol; independent 0.5 thresholds)

Do not compare the leaderboard score directly with argmax macro-F1.

## Repository structure
- `notebooks/`: exploratory analysis, baselines, and deep-learning experiments
- `src/predict.py`: single-image prediction helper
- `src/make_submission.py`: generates a submission CSV from metadata and images
- `data/processed/`: processed lesion/image index (raw images are not included)
- `splits/`: lesion-level train/validation/test splits
- `reports/`: final report, model card, experiment outputs, subgroup analysis
- `artifacts/`, `figures/`: data audits and visualisations
- `submission.csv`: recorded leaderboard submission

## Environment
Use Python 3.10 or 3.11 with PyTorch, torchvision, pandas, numpy, scikit-learn and Pillow. Install compatible PyTorch/torchvision builds for your operating system and accelerator.

```bash
pip install torch torchvision pandas numpy scikit-learn pillow
```

## Data and checkpoint setup
Raw MILK10k images and the model checkpoint are intentionally not committed to GitHub because of size and dataset licensing.

1. Place the dermoscopic image files in a local directory, named `{ISIC_ID}.jpg` (or `.jpeg`).
2. Place the selected checkpoint at `checkpoints/resnet18_moderate_best.pth`.
3. Keep the dataset metadata CSV available locally. Do not commit raw images or restricted data.

## Single-image inference
From the repository root:

```bash
python src/predict.py path/to/ISIC_image.jpg --checkpoint checkpoints/resnet18_moderate_best.pth
```

The function `predict(image_path, checkpoint_path=...)` returns the predicted class, confidence, and probabilities for all 11 classes.

## Generate a submission
The metadata CSV must contain `lesion_id`, `isic_id`, and `image_type`. The script selects dermoscopic rows and expects image files in the supplied image directory.

```bash
python src/make_submission.py \
  --metadata path/to/test_metadata.csv \
  --image-dir path/to/test_images \
  --checkpoint checkpoints/resnet18_moderate_best.pth \
  --output submission_generated.csv
```

It validates unique lesion IDs, row count, and probability sums. Compare the generated file with the course's exact submission schema before uploading.

## Reproducibility and limitations
The project uses lesion-level splits (60/20/20, seed 42). The checkpoint and raw image files are not included in this repository snapshot, so inference requires local copies. The subgroup analysis currently retained for sex, age, site, and skin tone is based on the original ResNet18 baseline predictions, not the selected moderate-sampling model. See `reports/final_report.md` and `reports/model_card.md` for the distinction and limitations.

This model is for coursework only, not clinical use.


## Final evaluation results

The selected ResNet18 moderate-sampling checkpoint achieved validation macro-F1 **0.3812** and held-out test argmax macro-F1 **0.3702** on the original lesion-level test split (1,048 lesions). Test accuracy was **0.6498**. The leaderboard-style fixed 0.5 one-vs-rest macro-F1 was **0.3161**. The official ISIC blind leaderboard score was **0.344** on 479 lesions; it is a separate evaluation.

Detailed reproducible outputs are in `reports/final_test_evaluation/`, including per-lesion predictions, classification report, confusion matrix, misclassified-image gallery, fairness analysis by sex and skin-tone class, updated scoreboard, and evaluation manifest. The original test split is defined in `data/processed/image_dataset.csv` and `splits/lesion_level_split.csv`.

Run `src/predict.py` for single-image inference and `src/make_submission.py` to reproduce the leaderboard-format submission. See the model card for intended use and limitations.
