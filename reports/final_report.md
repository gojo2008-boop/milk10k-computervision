# MILK10K Skin Lesion Classification: Final Project Report

## 1. Problem and dataset

This project studies 11-class classification of dermoscopic skin-lesion images from MILK10K. The labelled dataset contains 5,240 unique lesions. The target classes, in model-output order, are:

`AKIEC, BCC, BEN_OTH, BKL, DF, INF, MAL_OTH, MEL, NV, SCCKA, VASC`.

The dataset is strongly imbalanced. The rarest classes contain very few lesions, which makes their estimates unstable and makes macro-averaged metrics more informative than accuracy alone. One dermoscopic image per lesion was used for the reported image-classification experiments.

The supplementary metadata contains demographic and acquisition attributes, including age, sex, `skin_tone_class`, and anatomical site. These attributes are used for subgroup analysis, not as model inputs in the image-only classifier.

## 2. Data splitting and leakage control

The project uses a lesion-level split with 60% training, 20% validation, and 20% test data (3,144 / 1,048 / 1,048 lesions), using random seed 42. Splitting by lesion rather than by image is important because multiple images can belong to the same lesion. This avoids placing images of the same lesion in both training and evaluation partitions.

The test set is held out for final evaluation. Model and training-strategy selection should be based on validation performance, not repeated test-set tuning.

## 3. Models and experimental results

### Classical machine-learning baselines

The strongest reported classical test result was combined-feature Logistic Regression (test macro-F1 0.2335). The best cross-validation mean was obtained by combined-feature SVM (mean macro-F1 0.2273). These handcrafted-feature baselines were weaker than the deep-learning models on the reported test macro-F1.

### Deep-learning comparison

The original notebook reports the following results:

| Model / experiment | Validation argmax macro-F1 | Test argmax macro-F1 |
|---|---:|---:|
| Original ResNet18 | 0.3658 | 0.3319 |
| ResNet18 with moderate class sampling | **0.3812** | **0.3702** |
| ResNet18 with augmentation | 0.3722 | Not established in retained outputs |
| Further fine-tuned ResNet18 | 0.3742 | 0.3262 |
| Frozen ViT-B/16 | 0.3127 | 0.3210 |

The moderate-sampling ResNet18 is the strongest of the retained experiments by validation and reported local test argmax macro-F1. Its validation score is stored in the supplied checkpoint metadata (0.381241, best epoch 5). The notebook records local test argmax macro-F1 of 0.3702 and a class-level report for this model.

The ViT comparison is informative but not a controlled architecture-only comparison: the ResNet18 and ViT used different fine-tuning strategies. The original ResNet18 and ViT results are retained as the baseline comparison. The moderate-sampling result is reported as a subsequent ResNet18 training-strategy experiment.

**Selection:** ResNet18 with moderate class sampling (`resnet18_moderate_best.pth`).

## 4. Evaluation metric and leaderboard

Two different evaluation conventions appear in the project and must not be conflated:

1. **Local multiclass macro-F1:** the predicted class is `argmax` over the 11 softmax probabilities, then macro-F1 is calculated across the 11 classes. The reported moderate-sampling test result is 0.3702 under this convention.
2. **Leaderboard metric:** each class probability is independently thresholded at 0.5 and one-vs-rest F1 is calculated for each class, followed by the unweighted mean across classes. This is not the same as argmax multiclass macro-F1.

The submitted leaderboard CSV contains 479 unique lesions and 11 class-probability columns. The recorded leaderboard submission completed successfully with an official score of 0.344. This score should be reported separately from the local argmax test macro-F1 because the evaluation protocols and/or evaluated examples differ. The displayed leaderboard validation value of 1.000 is not interpreted as a model validation metric.

The notebook also explores class-specific thresholds on the validation set. Those thresholds are an experimental analysis and must not be substituted for the leaderboard's fixed 0.5 threshold.

## 5. Error analysis

The retained class-level report for the moderate-sampling model shows substantial variation across classes:

| Class | Test support | F1 |
|---|---:|---:|
| AKIEC | 61 | 0.33 |
| BCC | 504 | 0.82 |
| BEN_OTH | 9 | 0.00 |
| BKL | 109 | 0.25 |
| DF | 10 | 0.47 |
| INF | 10 | 0.14 |
| MAL_OTH | 2 | 0.00 |
| MEL | 90 | 0.41 |
| NV | 149 | 0.68 |
| SCCKA | 95 | 0.54 |
| VASC | 9 | 0.43 |

The model performs best on BCC and NV, which have substantially more training examples. It has no correctly identified examples for BEN_OTH and MAL_OTH in this test split. These results are consistent with the difficulty of learning rare categories from limited data.

The selected moderate-sampling model's final confusion matrix and ten actual high-confidence misclassified dermoscopic images are now included in `reports/final_test_evaluation/`. These were generated by running the saved checkpoint on the exact 1,048 lesions in the original held-out test split. The older confusion counts and original ResNet18 prediction file remain clearly labelled as baseline artifacts.

## 6. Fairness and subgroup analysis

### Existing subgroup results

The repository's original subgroup tables are based on the original ResNet18 test predictions, not the selected moderate-sampling model. They are retained as baseline subgroup analysis:

| Attribute | Group | n | Macro-F1 |
|---|---|---:|---:|
| Sex | Male | 636 | 0.2942 |
| Sex | Female | 412 | 0.3911 |
| Age | 0–29 | 32 | 0.1905 |
| Age | 30–49 | 149 | 0.3100 |
| Age | 50–69 | 446 | 0.3965 |
| Age | 70+ | 416 | 0.2698 |
| Site | Head/neck | 288 | 0.2898 |
| Site | Lower extremity | 218 | 0.3128 |
| Site | Upper extremity | 157 | 0.3696 |

### Skin-tone metadata analysis

The supplementary `training_input.csv` metadata was joined to the original ResNet18 test-prediction file by `lesion_id`. The metadata contains one consistent `skin_tone_class` value per lesion. This produced the following baseline subgroup results:

| Skin-tone class | n | Accuracy | Balanced accuracy | Macro-F1 |
|---:|---:|---:|---:|---:|
| 0 | 2 | 0.5000 | 0.5000 | 0.0909 |
| 1 | 17 | 0.5882 | 0.4833 | 0.2684 |
| 2 | 113 | 0.5752 | 0.4224 | 0.3643 |
| 3 | 614 | 0.5570 | 0.3771 | 0.3241 |
| 4 | 223 | 0.5471 | 0.3327 | 0.2538 |
| 5 | 79 | 0.4557 | 0.2260 | 0.1539 |

These skin-tone results are also for the original ResNet18 baseline because the retained test-prediction file contains predictions from that model. They are not presented as fairness results for the selected moderate-sampling model. In particular, the subgroup with skin-tone class 0 has only two test observations, so its score is not statistically meaningful. Small subgroup sizes and class imbalance make all subgroup macro-F1 values uncertain.

**Selected-model subgroup analysis:** the final moderate-sampling checkpoint was evaluated on the 1,048 held-out test lesions and joined to the supplied `training_input.csv` metadata by `lesion_id`. The resulting subgroup macro-F1 values for sex and skin-tone class are included in `reports/final_test_evaluation/fairness_by_skin_tone_and_sex.csv`. These are descriptive estimates, not proof of equal performance. Skin-tone classes 0 (n=2) and 1 (n=17) are especially small and unstable. Age and site subgroup metrics are not claimed for this selected checkpoint.

## 7. Limitations, intended use, and conclusion

This is an educational coursework experiment, not a clinical diagnostic system. It must not be used for diagnosis, screening, triage, or treatment decisions. The model was evaluated on one held-out split from the same dataset source; no external validation was performed. Class imbalance, very small rare-class supports, limited image resolution, and subgroup sample sizes constrain interpretation. Softmax outputs are not calibrated clinical risks.

The selected model is the moderate-sampling ResNet18 checkpoint, based on its best validation macro-F1 (0.3812) and recorded local test argmax macro-F1 (0.3702). The official leaderboard score (0.344) is reported separately under the leaderboard's independent 0.5 threshold metric. The model and all analyses are intended solely for non-commercial educational/research use under the dataset's applicable licence.

## References

- MILK10k dataset, DOI: 10.34970/262082.
- MILK10k associated publication, DOI: 10.1016/j.jid.2025.06.1594.
- ImageNet-pretrained ResNet18 weights are provided by torchvision; cite the relevant torchvision / ImageNet sources in the submitted bibliography if required by the course.


## Final held-out test evaluation update

The final test evaluation was rerun using the original repository split file (`data/processed/image_dataset.csv`, `split == "test"`), not the separate metadata file previously used for image extraction. The extracted archive was verified to contain exactly the 1,048 expected image IDs, with no missing or extra images.

The selected moderate-sampling ResNet18 checkpoint (best validation macro-F1 0.3812, epoch 5) achieved **test argmax macro-F1 0.3702**, **accuracy 0.6498**, and **fixed 0.5 one-vs-rest macro-F1 0.3161** on this held-out test set. The argmax macro-F1 is the local multiclass comparison metric; the fixed-threshold metric is reported separately because it follows the leaderboard-style scoring rule.

The updated scoreboard, per-class report, confusion matrix, ten-image error gallery, per-lesion probabilities, and selected-model fairness analysis are saved in `reports/final_test_evaluation/`. The official blind leaderboard score remains 0.344 on 479 lesions and must not be conflated with the local held-out test results.
