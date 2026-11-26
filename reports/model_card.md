# Model Card: MILK10K Moderate-Sampling ResNet18

## Model details
- Architecture: ResNet18, ImageNet-pretrained initialisation
- Selected checkpoint: `checkpoints/resnet18_moderate_best.pth`
- Model identifier in checkpoint: `resnet18_moderate`
- Number of classes: 11
- Input: RGB dermoscopic image, resized to 128 × 128
- Framework: PyTorch / torchvision
- Best validation argmax macro-F1: 0.381241 (epoch 5)
- Recorded local test argmax macro-F1: 0.3702

## Intended use
Educational and non-commercial research only. Not for clinical diagnosis, screening, triage, treatment recommendations, or use as a medical device.

## Data
MILK10k, 5,240 labelled lesions. A lesion-level 60/20/20 train/validation/test split was used (3,144 / 1,048 / 1,048; seed 42). One dermoscopic image per lesion was used in the reported experiments. The target order is:
`AKIEC, BCC, BEN_OTH, BKL, DF, INF, MAL_OTH, MEL, NV, SCCKA, VASC`.

## Training
The retained notebook describes a ResNet18 with the classification head trained using a moderate class-sampling strategy for five epochs. The supplied checkpoint metadata confirms model name, class count, image size, best validation macro-F1 and epoch. The model was selected by validation macro-F1.

## Evaluation
- Validation argmax macro-F1: 0.3812
- Local held-out test argmax macro-F1: 0.3702 (recorded in notebook output)
- Official leaderboard score: 0.344 (separate evaluation protocol using independent 0.5 class-probability thresholds)

Do not compare the leaderboard score directly with argmax macro-F1 as if they were the same metric.

## Known performance limitations
The retained moderate-model test classification report shows F1 of 0.82 for BCC and 0.68 for NV, while BEN_OTH and MAL_OTH have F1 of 0.00 on this split. The rare classes have very small supports. Predictions may be unreliable, especially for rare categories.

## Fairness
Sex, age, anatomical-site, and newly computed skin-tone subgroup metrics available in the repository are based on the original ResNet18 baseline predictions, not the selected moderate-sampling checkpoint. They are exploratory baseline analyses and must not be interpreted as fairness validation of this model. Subgroup evaluation for the selected checkpoint remains incomplete.

## Limitations and ethical considerations
- One train/validation/test split only.
- No external dataset validation.
- Strong class imbalance and very small rare-class sample sizes.
- 128 × 128 input resolution may discard clinically relevant detail.
- No probability calibration or clinical validation.
- Subgroup metrics are limited by small samples and are not evidence of clinical fairness.

## Reproducibility
The repository archive does not include the large checkpoint or raw image files. Place the checkpoint at `checkpoints/resnet18_moderate_best.pth` and configure the local image root as described in the README. The prediction script loads the checkpoint and returns all 11 softmax probabilities.


## Final evaluation update

- Selected checkpoint: ResNet18 with moderate class sampling, epoch 5.
- Validation macro-F1: 0.3812.
- Held-out test argmax macro-F1: 0.3702 (1,048 lesions).
- Held-out test accuracy: 0.6498.
- Held-out test fixed-threshold (0.5) one-vs-rest macro-F1: 0.3161.
- Official ISIC leaderboard score: 0.344 (479 blind lesions; separate evaluation).
- Selected-model subgroup macro-F1 by sex and `skin_tone_class` is provided in `reports/final_test_evaluation/fairness_by_skin_tone_and_sex.csv`.

This model is an educational research prototype only. It is not intended for clinical diagnosis, screening, triage, or treatment decisions. Rare classes and small demographic subgroups have limited support, and no external clinical validation has been performed.
