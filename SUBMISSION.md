# MILK10k Project Submission

## Milestone 1 Overview

This repository contains the Milestone 1 implementation for the MILK10k computer vision classification project.

The milestone establishes a reproducible, lesion-level data pipeline for 11-class skin-lesion classification. No predictive model training is included in this milestone.

## Dataset

MILK10k contains:

* 10,480 images
* 5,240 unique lesions
* 2 images per lesion
* 11 diagnostic classes
* One dermoscopic image and one clinical close-up per lesion

The raw image dataset is stored locally and is **not committed to GitHub**.

The local dataset location is configured in `src/config.py`.

## Completed Components

### Dataset Integrity

The dataset was audited for:

* unique image IDs
* unique lesion IDs
* expected image counts per lesion
* presence of both image types
* metadata and ground-truth linkage
* one-hot label consistency
* missing image files
* unexpected image files

### Exploratory Data Analysis

The following distributions were examined:

* diagnostic class distribution
* image type
* age
* sex
* anatomical site
* metadata missingness

### Label Strategy

The one-hot ground-truth representation was converted into a single diagnostic label for each lesion across the 11 target classes.

A lesion-level table was created containing the lesion ID, diagnostic label, clinical image ID, and dermoscopic image ID.

### Leakage-Safe Splitting

Train, validation, and test splits were created at the lesion level.

Stratification was performed using the diagnostic class while ensuring that images belonging to the same lesion cannot appear in different splits.

The resulting split files are stored in `splits/`.

### Image Preprocessing

Images are resized to 224 × 224 pixels, converted to RGB tensors, and normalized using ImageNet mean and standard deviation values.

Training uses conservative augmentation including:

* horizontal flipping
* small rotations
* mild brightness changes
* small hue changes

Validation and test preprocessing is deterministic.

### PyTorch Data Pipeline

A custom `LesionDataset` was implemented to load images and labels while preserving lesion and image identifiers.

Training, validation, and test DataLoaders were created with appropriate transformations.

### Class Imbalance

Balanced class weights were calculated using the training set only.

A `WeightedRandomSampler` was implemented to increase the sampling probability of underrepresented classes during training.

Validation and test sets are not resampled.

### Sanity Checks

The final pipeline was checked for:

* lesion leakage
* image overlap
* valid class mappings
* expected tensor dimensions
* valid tensor values
* deterministic evaluation preprocessing
* valid sampler configuration

## Reproducibility

The project uses a fixed random seed of 42 for dataset splitting and sampling.

Python dependencies are listed in `requirements.txt`.

Raw images are intentionally excluded from version control through `.gitignore`.

Install the required dependencies with:

```text
pip install -r requirements.txt
```

Then open:

```text
notebooks/part_b.ipynb
```

The notebook expects the MILK10k dataset to be available at the path specified in `src/config.py`.

## Key Artifacts

Important generated artifacts include:

```text
artifacts/
├── dataset_integrity_audit.csv
├── lesion_table.csv
├── image_quality_report.csv
├── class_distribution.csv
└── pipeline_sanity_checks.csv
```

Figures are stored in:

```text
figures/
```

The final lesion-level splits are stored in:

```text
splits/
├── train.csv
├── val.csv
└── test.csv
```

The milestone report is stored at:

```text
reports/milestone_report.md
```

## Scope and Limitations

This milestone focuses on data preparation, exploratory analysis, leakage prevention, preprocessing, class balancing, and the PyTorch input pipeline.

No diagnostic model is trained or evaluated as part of this milestone. Therefore, no claims about classification accuracy, generalization, or clinical performance are made.

The strong class imbalance and the presence of multiple images per lesion are important considerations for subsequent model development.

## Next Milestone

The next stage will use the completed data pipeline for model training, validation, lesion-level prediction aggregation, and evaluation using appropriate multiclass metrics.
