# MILK10K Skin Lesion Classification
## Computer Vision Final Project

### Abstract

This project investigates multi-class skin lesion classification using
the MILK10K dataset. Classical machine learning and deep learning
approaches were evaluated using a lesion-level training, validation and
test split. The experiments included handcrafted image features,
fine-tuned ResNet18 and a frozen-backbone Vision Transformer (ViT-B/16).

ResNet18 achieved the highest held-out test Macro-F1 score of 0.3319.
The results demonstrate the benefits of deep image representations
compared with the handcrafted-feature baselines, while also revealing
challenges caused by class imbalance and visually similar lesion types.

---

## 1. Introduction

Skin lesion classification is a challenging computer vision task.
Different lesion categories can share visual characteristics, while
some diagnostic categories are represented by very few examples.

The objective of this project was to develop and evaluate image
classification pipelines capable of predicting one of 11 lesion
categories from dermoscopic images.

The project compares classical machine learning approaches with
deep learning architectures and investigates model errors and
subgroup performance.

## 2. Dataset

The MILK10K dataset contains 5,240 unique lesions and 10,480 images.
Each lesion has two associated image types.

For this experiment, one dermoscopic image was selected per lesion,
giving 5,240 images.

The task contains 11 classes:

AKIEC, BCC, BEN_OTH, BKL, DF, INF, MAL_OTH, MEL, NV, SCCKA and VASC.

The dataset is substantially imbalanced. BCC is the most represented
class, while MAL_OTH contains only nine lesions.

### 2.1 Data splitting

The dataset was divided at lesion level using random seed 42.

| Split | Lesions | Percentage |
|---|---:|---:|
| Training | 3,144 | 60% |
| Validation | 1,048 | 20% |
| Test | 1,048 | 20% |

Lesion-level splitting prevents images from the same lesion appearing
in multiple splits, reducing the risk of data leakage.

All experiments used the same split.

## 3. Methodology

### 3.1 Data preprocessing

Dermoscopic images were resized to 128 x 128 pixels and converted
to RGB tensors.

ImageNet mean and standard deviation were used for normalisation.

Random horizontal flipping was used as training augmentation for
the deep learning experiments.

### 3.2 Classical machine learning

Handcrafted features were extracted from the images, including
colour statistics, colour histogram information and texture or
edge-related features.

Three feature configurations were evaluated:

- Colour features
- Texture and edge features
- Combined features

The following classifiers were tested:

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest

Classical model performance was evaluated using cross-validation
and the held-out test set.

### 3.3 ResNet18

A ResNet18 model pretrained on ImageNet was used.

The backbone was initially frozen, with the final residual block
(layer4) and classification head unfrozen for fine-tuning.

Training configuration:

| Parameter | Setting |
|---|---|
| Optimiser | AdamW |
| Learning rate | 0.0001 |
| Batch size | 32 |
| Epochs | 5 |
| Loss | Class-weighted Cross Entropy |
| Trainable components | layer4 and classification head |

The checkpoint with the highest validation Macro-F1 was selected.

### 3.4 Vision Transformer

A pretrained ViT-B/16 architecture was evaluated.

The backbone was frozen and a new classification head was trained
for the 11-class task.

The model used the same image resolution and dataset split as
ResNet18.

This experiment evaluates a frozen-feature transformer approach
rather than full end-to-end fine-tuning.

## 4. Experimental Results

### 4.1 Classical machine learning

The strongest classical result on the held-out test set was
Combined-feature Logistic Regression, with a Macro-F1 of 0.2335.

The best cross-validation Macro-F1 mean was achieved by the
Combined-feature SVM, at 0.2273.

Random Forest achieved relatively high accuracy in some experiments,
but its Macro-F1 and balanced accuracy were considerably lower.
This illustrates why accuracy alone is insufficient for an
imbalanced classification task.

### 4.2 Deep learning comparison

| Metric | ResNet18 | ViT-B/16 |
|---|---:|---:|
| Validation Macro-F1 | 0.3658 | 0.3127 |
| Test Macro-F1 | 0.3319 | 0.3210 |
| Test accuracy | 0.5496 | 0.4857 |
| Balanced accuracy | 0.3615 | 0.3785 |
| Test loss | 1.9060 | 1.8166 |

ResNet18 achieved the highest test Macro-F1 and test accuracy.
ViT-B/16 achieved slightly higher balanced accuracy and lower
test loss.

The comparison is not a pure architecture-only comparison because
the models used different training strategies. ResNet18 fine-tuned
layer4 and its classification head, whereas ViT-B/16 used a frozen
backbone and trained classification head.

### 4.3 Overall model comparison

The deep learning models outperformed the classical approaches
on held-out test Macro-F1.

ResNet18 was selected as the final model because Macro-F1 was the
primary evaluation metric and it achieved the highest score.

## 5. Error Analysis

The ResNet18 confusion matrix revealed several recurring errors.

Important confusions included:

- BCC predicted as SCCKA: 54 cases
- BCC predicted as AKIEC: 50 cases
- BCC predicted as BKL: 34 cases
- MEL predicted as NV: 29 cases
- NV predicted as MEL: 26 cases
- BKL predicted as SCCKA: 25 cases

These errors suggest that several lesion categories have overlapping
visual characteristics.

The model also struggled with underrepresented categories.
BEN_OTH and MAL_OTH had zero recall on the held-out test set.

The small number of examples in these categories makes it difficult
for the model to learn robust class-specific representations.

## 6. Fairness and Subgroup Analysis

The final ResNet18 model was evaluated across available demographic
and anatomical subgroups.

### 6.1 Sex

| Group | Samples | Macro-F1 |
|---|---:|---:|
| Female | 412 | 0.3911 |
| Male | 636 | 0.2942 |

### 6.2 Age

| Group | Samples | Macro-F1 |
|---|---:|---:|
| 0-29 | 32 | 0.1905 |
| 30-49 | 149 | 0.3100 |
| 50-69 | 446 | 0.3965 |
| 70+ | 416 | 0.2698 |

### 6.3 Anatomical site

| Group | Samples | Macro-F1 |
|---|---:|---:|
| Head/neck | 288 | 0.2898 |
| Lower extremity | 218 | 0.3128 |
| Upper extremity | 157 | 0.3696 |

Subgroup performance varied across the available groups.

These differences should not be interpreted as definitive evidence
of demographic bias. Differences in class composition, subgroup
sample sizes and missing metadata may influence the results.

Age was missing for five test samples, while anatomical site was
missing for 385 test samples.

## 7. Limitations

Several limitations affect the experiments:

1. The dataset has substantial class imbalance.
2. Some classes contain very few examples.
3. The experiments use one lesion-level random split.
4. Images were resized to 128 x 128 pixels, potentially removing
   fine visual detail.
5. No external dataset validation was performed.
6. The ResNet18 and ViT experiments used different fine-tuning
   strategies.
7. Subgroup analysis was limited by missing metadata and small
   group sizes.
8. Model confidence scores were not calibrated as clinical risk
   estimates.

## 8. Conclusion

This project compared handcrafted-feature machine learning
approaches with two deep learning architectures for 11-class
skin lesion classification.

ResNet18 achieved the strongest held-out test Macro-F1 score
of 0.3319, outperforming the evaluated classical approaches
and ViT-B/16 on the primary metric.

However, the results also demonstrate that class imbalance,
rare diagnostic categories and visual similarity remain
significant challenges.

Future work could investigate higher image resolutions,
stronger data augmentation, improved class balancing,
more extensive fine-tuning and external validation.

The final model is intended strictly for educational and
research purposes, not clinical diagnosis.

## 9. Reproducibility and Project Files

The project repository includes the training notebook,
evaluation outputs, model checkpoints, prediction function,
model card and experiment scoreboards.

The final ResNet18 prediction interface is:

`predict(image_path)`

It returns a predicted class, confidence score and probabilities
for all 11 classes in the canonical class order.

---

## Appendix: Final Model

**Selected model:** ResNet18

**Validation Macro-F1:** 0.3658

**Test Macro-F1:** 0.3319

**Test accuracy:** 0.5496

**Test balanced accuracy:** 0.3615

**Checkpoint:** `checkpoints/resnet18_best.pth`