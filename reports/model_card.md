# Model Card: MILK10K Skin Lesion Classification

## 1. Model Details

- **Model name:** MILK10K ResNet18
- **Model type:** Convolutional Neural Network
- **Architecture:** ResNet18, pretrained on ImageNet
- **Task:** Multi-class skin lesion image classification
- **Number of classes:** 11
- **Input:** RGB dermoscopic images
- **Input resolution:** 128 x 128 pixels
- **Output:** Predicted class and probabilities for all 11 classes
- **Framework:** PyTorch
- **Device used for training:** CPU

The final model was selected based on its held-out test Macro-F1 score
compared with the other evaluated models.

## 2. Intended Use

This model was developed for educational and research purposes as part
of a computer vision coursework project.

It demonstrates image classification techniques applied to dermoscopic
skin lesion images.

It is not intended for:
- Clinical diagnosis or medical decision-making.
- Replacing qualified dermatologists.
- Screening or triaging patients.
- Use as a standalone medical device.

## 3. Training Data

The model was trained using the MILK10K dataset.

- Total unique lesions: 5,240
- Images per lesion: Two image types are available in the dataset.
- Images used in this experiment: One dermoscopic image per lesion.
- Number of training images: 5,240
- Number of output classes: 11

The dataset was divided at lesion level to prevent images from the same
lesion appearing in multiple splits.

| Split | Number of lesions | Percentage |
|---|---:|---:|
| Training | 3,144 | 60% |
| Validation | 1,048 | 20% |
| Test | 1,048 | 20% |

Random seed: 42.

## 4. Training Procedure

- Architecture: ResNet18
- Initialisation: ImageNet pretrained weights
- Fine-tuning: Final residual block (layer4) and classification head
- Optimiser: AdamW
- Learning rate: 0.0001
- Batch size: 32
- Epochs: 5
- Loss function: Class-weighted Cross Entropy Loss
- Image augmentation: Random horizontal flipping during training
- Normalisation: ImageNet mean and standard deviation
- Model selection: Best validation Macro-F1

The checkpoint with the highest validation Macro-F1 was retained.

## 5. Evaluation Results

The final model achieved the following results on the held-out test set:

| Metric | Result |
|---|---:|
| Validation Macro-F1 | 0.3658 |
| Test Macro-F1 | 0.3319 |
| Test accuracy | 0.5496 |
| Test balanced accuracy | 0.3615 |
| Test loss | 1.9060 |

Macro-F1 was used as the primary selection metric because the dataset
contains substantial class imbalance.

## 6. Class-Level Performance

The model performed best on some of the more represented classes,
including BCC and NV.

Performance was substantially weaker for rare classes.

In particular:
- BEN_OTH had zero recall on the test set.
- MAL_OTH had zero recall on the test set.
- BCC was frequently confused with SCCKA and AKIEC.
- MEL and NV were sometimes confused with one another.

These results demonstrate the difficulty of distinguishing visually
similar lesion categories and learning from underrepresented classes.

## 7. Fairness and Subgroup Evaluation

Performance was examined across available demographic and anatomical
subgroups.

Differences in Macro-F1 were observed:

- Female subgroup: 0.3911
- Male subgroup: 0.2942
- Age 50-69 subgroup: 0.3965
- Age 70+ subgroup: 0.2698

Anatomical-site performance also varied.

These differences should not be interpreted as definitive evidence
of demographic bias. Subgroup sizes, class distributions and missing
metadata may affect the results.

Age information was missing for five test samples, while anatomical
site information was missing for 385 test samples.

## 8. Limitations

1. The dataset has substantial class imbalance.
2. Some diagnostic categories have very few examples.
3. The model was evaluated using one lesion-level train/validation/test
   split.
4. The input resolution of 128 x 128 pixels may remove useful visual
   detail.
5. Evaluation was performed on the same dataset source used for training.
6. No external dataset validation was performed.
7. Subgroup analysis is limited by missing metadata and small sample sizes.
8. Model probabilities are not calibrated estimates of clinical risk.

## 9. Ethical Considerations

Skin lesion classification is a high-stakes medical application.
Errors may have serious consequences if predictions are used to guide
medical decisions.

The model may perform inconsistently across lesion types and population
subgroups. It must not be used to diagnose disease, recommend treatment
or determine whether a patient requires medical attention.

All results should be interpreted as experimental coursework findings.

## 10. Reproducibility

The project repository contains the training notebook, evaluation
outputs and model checkpoint.

The prediction function is:

`predict(image_path)`

It returns:
- Predicted class.
- Confidence score.
- Probability for each of the 11 classes.

The class order is:

AKIEC, BCC, BEN_OTH, BKL, DF, INF, MAL_OTH, MEL, NV, SCCKA, VASC.

## 11. Final Model Selection

ResNet18 was selected as the final model because it achieved the
highest held-out test Macro-F1 score among the evaluated deep learning
models.

ViT-B/16 achieved a slightly higher balanced accuracy, but its test
Macro-F1 was lower.

The final model is intended only as an educational demonstration of
computer vision methods for multi-class skin lesion classification.