# MILK10k Computer Vision Project
## Milestone 1 Report

### 1. Dataset and Problem Definition

MILK10k contains 10,480 dermoscopic and clinical close-up images representing
5,240 unique lesions. Each lesion has exactly two associated images: one
dermoscopic image and one clinical close-up image. The diagnostic target is an
11-class classification problem using the labels AKIEC, BCC, BEN_OTH, BKL, DF,
INF, MAL_OTH, MEL, NV, SCCKA, and VASC.

The dataset is substantially imbalanced. BCC is the largest class, while
MAL_OTH, BEN_OTH, VASC, INF, and DF are much smaller. Because multiple images
belong to the same lesion, lesion identity must be treated as the grouping
unit throughout dataset splitting and evaluation.

### 2. Dataset Integrity and Exploratory Analysis

The integrity audit found 10,480 unique image IDs and 5,240 unique lesion IDs.
Every lesion has exactly two images, with one dermoscopic image and one clinical
close-up. All ground-truth lesions could be linked to metadata, and every
lesion had exactly one positive diagnostic label. All 10,480 expected JPEG
files were present, with no missing or unexpected image files.

Metadata analysis showed variation in age, sex, anatomical site, image
manipulation, and diagnostic confirmation type. Missing metadata was retained
rather than being treated as an image-level failure. Image quality statistics
were collected for width, height, mean pixel intensity, pixel standard
deviation, and file size and saved as a reproducible artifact.

The class distribution and key metadata distributions were visualized and
saved under the figures directory.

### 3. Label Strategy and Data Splitting

The one-hot ground-truth representation was converted into a single
11-class diagnostic label for each lesion. A lesion-level table was created
containing the lesion ID, diagnostic label, dermoscopic image ID, and clinical
image ID.

The final train, validation, and test sets were created at the lesion level
using stratification by diagnostic class. This prevents the two images from
the same lesion from appearing in different splits. The resulting split
contains approximately 60% of lesions for training and 20% each for validation
and testing, corresponding to approximately 6,288 training images, 2,096
validation images, and 2,096 test images.

No lesion overlap exists between the three splits, and all original images are
assigned to exactly one split. The split CSV files are stored in the splits
directory.

### 4. Preprocessing and Augmentation

Images are resized to 224 × 224 pixels, converted to RGB tensors, and
normalized using ImageNet channel statistics. Training images use conservative
augmentation consisting of horizontal flipping, small rotations, and mild
brightness and hue adjustments.

Large color or brightness changes were avoided because pigmentation,
erythema, and skin tone can contain clinically relevant information. Validation
and test images use deterministic resizing, tensor conversion, and
normalization without random augmentation.

Repeated application of the evaluation transform to the same image produced
identical tensors, confirming deterministic evaluation preprocessing.

### 5. PyTorch Data Pipeline

A custom PyTorch Dataset was implemented to load images from disk, apply the
appropriate transformation, convert diagnostic labels to integer class
indices, and return image IDs and lesion IDs alongside the image and label.

Separate DataLoaders were created for training, validation, and testing.
Training uses a weighted sampler to address class imbalance, while validation
and test sets are not resampled.

Balanced class weights were calculated from the training set only. These
weights were assigned to individual training samples and used to construct a
WeightedRandomSampler with replacement. Inspection of the sampled distribution
confirmed that the sampler increases the representation of underrepresented
classes.

### 6. Validation and Reproducibility

Pipeline sanity checks verified dataset sizes, image and lesion uniqueness,
class mappings, split isolation, tensor dimensions, numerical validity,
evaluation determinism, and sampler configuration.

All raw images remain outside the repository and are excluded through
.gitignore. Reproducible split CSV files, figures, quality reports, and
sanity-check artifacts are stored in the repository. The project uses a fixed
random seed of 42 for reproducible splitting and sampling.

### 7. Limitations and Next Steps

The main limitation at this milestone is the strong class imbalance,
particularly for the rare diagnostic categories. The dataset also contains
two images per lesion, making lesion-level grouping essential to prevent
information leakage.

No predictive model is trained or evaluated in this milestone. The completed
work establishes a leakage-safe dataset, preprocessing pipeline, class
balancing strategy, and reproducible PyTorch data pipeline. The next stage
will use this infrastructure for model training and lesion-level evaluation.