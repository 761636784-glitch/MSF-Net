# MSF-Net

**Morphology–Spectral Fusion for Differentiating Early Hepatocellular Carcinoma From High-Grade Dysplastic Nodules**

This repository provides the core PyTorch implementation of **MSF-Net**, a morphology–spectral fusion framework for distinguishing early hepatocellular carcinoma (eHCC) from high-grade dysplastic nodules (HGDN) using microscopic hyperspectral images (HSI).

MSF-Net extracts morphological and spectral information from the **same hyperspectral cube**. Its morphology branch processes HSI-derived pseudo-RGB images, while its spectral branch models regional spectral statistics and inter-band dependencies. An adaptive gated fusion module combines the two representations for classification.

## 1. Repository structure

```text
MSF-Net/
├── config.py          # Data paths and experiment settings
├── dataset.py         # ENVI loading and HSI preprocessing
├── model.py           # MSF-Net architecture
├── main.py            # Training, nested cross-validation and evaluation
├── requirements.txt   # Python dependencies
├── .gitignore
└── README.md
```

## 2. Method overview

### Hyperspectral input preparation

Each reflectance-calibrated HSI cube is divided into non-overlapping **100 × 100-pixel** patches. Two complementary inputs are derived from each patch:

- **Morphological representation:** pseudo-RGB image constructed using spectral bands nearest to **650, 550 and 450 nm**.
- **Spectral representation:** a **5 × 5** grid of **20 × 20-pixel** regions. The mean and standard-deviation spectra of each region are calculated, producing a **25 × 2 × 256** descriptor. Standard normal variate (SNV) normalization is applied to the regional mean spectra.

### Network architecture

- **Morphology branch:** ImageNet-pretrained SE-ResNet34 with a 256-dimensional morphological feature projection.
- **Spectral branch:** shared 1D-CNN–BiGRU encoder for regional spectral descriptors, followed by regional feature pooling into a 128-dimensional representation.
- **Adaptive fusion:** morphological and spectral features are projected to a common 128-dimensional space and combined using a learnable gating mechanism.
- **Classifier:** a fully connected classification head with dimensions **128 → 64 → 2**.

## 3. Installation

Python 3.10 or 3.11 is recommended.

```bash
pip install -r requirements.txt
```

The first training run uses ImageNet-pretrained ResNet34 weights provided through `torchvision`. An internet connection or a local torchvision weight cache may be required.

## 4. Data organization

Prepare reflectance-calibrated microscopic HSI data in ENVI format (`.hdr` and `.raw` file pairs):

```text
HSI_Reflectance/
├── ehcc/
│   ├── case01.hdr
│   ├── case01.raw
│   └── ...
└── hgdn/
    ├── case01.hdr
    ├── case01.raw
    └── ...
```

The ENVI headers should provide the spectral wavelength information. The default configuration is designed for **256 spectral channels** and a cohort containing **30 eHCC and 30 HGDN cases**. Inputs are expected to have undergone white- and dark-reference reflectance calibration before preprocessing.

Set the data root directory before running the scripts.

**Windows PowerShell:**

```powershell
$env:MSFNET_DATA_ROOT = "D:\HSI_Reflectance"
```

**Linux / macOS:**

```bash
export MSFNET_DATA_ROOT="/path/to/HSI_Reflectance"
```

## 5. Preprocessing

Run these commands from the repository root:

```bash
python dataset.py --scan
python dataset.py --prepare
```

The first command generates the patient manifest. The second performs patch construction, pseudo-RGB generation, regional spectral feature extraction and preprocessing. Generated files are stored under `MSFNET_DATA_ROOT/MSF-Net-Preprocessed/`.

## 6. Training and evaluation

The implementation uses **patient-grouped nested cross-validation**, with five outer folds and four inner folds. Model and decision-threshold selection are performed on the inner folds; performance is evaluated using held-out outer-fold predictions.

The default training settings are:

| Setting | Value |
|---|---|
| Batch size | 4 |
| Maximum epochs | 12 |
| Random seed | 2026 |
| Optimizer | AdamW |
| Backbone learning rate | 3 × 10⁻⁵ |
| Other modules learning rate | 5 × 10⁻⁵ |
| Weight decay | 1 × 10⁻⁴ |
| Loss function | Cross-entropy |
| Early-stopping patience | 4 |
| Bootstrap resamples | 1,000 |

**Model smoke test (CPU):**

```bash
python model.py --device cpu --batch-size 2
```

**Inspect patient-grouped cross-validation splits:**

```bash
python main.py --dry-run
```

**Run nested cross-validation:**

```bash
python main.py --nested --epochs 12 --batch-size 4 --bootstrap 1000
```

### Evaluation protocol

The positive class is **eHCC**. Patch-level performance is evaluated using AUC, accuracy, sensitivity, specificity and F1-score. Patient-level predictions are obtained by averaging the **top 20% of patch probabilities for eHCC**. Alternative aggregation methods (mean, median, maximum, and top 5%, 10% and 30%) are included for sensitivity analysis.

Confidence intervals are computed using **1,000 stratified patient-level bootstrap resamples**, preserving within-patient patch clustering. A fixed decision threshold of 0.5 is also evaluated as a sensitivity analysis.

## 7. Data availability

The clinical hyperspectral imaging data are not publicly available due to restrictions on clinical data sharing. Data supporting the study may be requested from the corresponding author, subject to applicable ethical and institutional requirements.

## 8. Citation

If you use this implementation, please cite the associated manuscript:

*MSF-Net: Morphology–Spectral Fusion for Differentiating Early Hepatocellular Carcinoma From High-Grade Dysplastic Nodules.*
