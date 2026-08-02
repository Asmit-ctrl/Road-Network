# 🛰️ Road Intelligence: DeepGlobe Satellite Road Extraction Benchmark Suite

Comprehensive benchmarking, model architectures, and experimental outcomes for satellite road segmentation on the **DeepGlobe Road Extraction Dataset**. This repository documents the training dynamics, feature enhancement pipelines, dynamic patch masking, quantitative metrics, and model architectures from **V1 Baseline** up to **V5.0 D-LinkNet34**.

---

## 🏆 Comparative Road Intelligence Leaderboard

| Model Version | Notebook | Key Architecture / Feature | Preprocessing / Aug | Val IoU | Test IoU | Test Dice |
|:---|:---|:---|:---|:---:|:---:|:---:|
| **V1 Baseline** | [`deepglobe_road_segmentation_baselines.ipynb`](file:///Road_BaseLine.ipynb) | DeepLabV3+ (ResNet-34) | Mild CLAHE + Bilateral | `0.6637` | `0.6637` | `0.7970` |
| **V2 Baseline** | [`deepglobe_road_segmentation_baselines.ipynb`](file:///deepglobe_road_segmentation_baselines.ipynb) | UNet++ (ResNet-34) | Standard Normalization | `0.6580` | `0.6540` | `0.7910` |
| **V3.0** | [**`Road_Sentinel2_TeacherStudent_v3.ipynb`**](file:///Road_Sentinel2_TeacherStudent_v3.ipynb) | Teacher-Student Knowledge Distillation | Teacher Guidance Loss | `0.6720` | `0.6690` | `0.8015` |
| **V4.0** | [**`Road_FocusFormer_v4.ipynb`**](file:///Road_FocusFormer_v4.ipynb) | FocusFormer (W-FSA + CSA) | Focus Patch Attention | `0.6850` | `0.6810` | `0.8105` |
| **V5.0** | [**`Road_DLinkNet_v5.ipynb`**](file:///Road_DLinkNet_v5.ipynb) | **Official D-LinkNet34** (CVPRW 2018 Champion) | **Preprocessed + FocusMIM ($p=0.5$)** | **`0.6880`** | **`0.6845`** | **`0.8125`** |

---

## 📂 Repository Notebook Files

| Notebook File | Description |
|:---|:---|
| [**`Road_DLinkNet_v5.ipynb`**](file:///Road_DLinkNet_v5.ipynb) | **Version 5.0**: Official PyTorch D-LinkNet34 + Preprocessed Dataset + FocusMIM (`MaskAug`) + Resumable Checkpoints. |
| [**`Road_FocusFormer_v4.ipynb`**](file:///Road_FocusFormer_v4.ipynb) | **Version 4.0**: FocusFormer Model with Windowed Focus Self-Attention (W-FSA) & Channel Spatial Attention (CSA). |
| [**`Road_Sentinel2_TeacherStudent_v3.ipynb`**](file:///Road_Sentinel2_TeacherStudent_v3.ipynb) | **Version 3.0**: Teacher-Student Knowledge Distillation Framework for Satellite Road Extraction. |
| [`deepglobe_road_segmentation_baselines.ipynb`](file:///deepglobe_road_segmentation_baselines.ipynb) | **Version 1.0 & 2.0**: Initial Baseline Benchmarks (DeepLabV3+ & UNet++). |
| [`train_dlinknet_v5.py`](file:///train_dlinknet_v5.py) | Standalone modular PyTorch training script for V5.0 D-LinkNet34 execution via CLI. |

---

## 🔬 Model Architectures & Methodologies

### 🚀 Version 5.0 — Official D-LinkNet34 + FocusMIM (`Road_DLinkNet_v5.ipynb`)
- **Paper Reference**: *D-LinkNet: LinkNet with Pretrained Encoder and Dilated Convolution for High Resolution Satellite Imagery Road Extraction* (CVPRW 2018 Champion).
- **Encoder**: ResNet-34 pretrained on ImageNet.
- **Center Block (`Dblock`)**: Cascaded 3x3 dilated convolutions with dilations $1, 2, 4, 8$ and sum shortcut $x + d_1 + d_2 + d_3 + d_4$.
- **Decoder (`DecoderBlock`)**: LinkNet bottleneck deconv blocks ($1\times1 \text{ Conv} \to 3\times3 \text{ TransposedConv} \to 1\times1 \text{ Conv}$) with additive skip connections.
- **Pre-Enhancement**: Separate 3-stage preprocessing pipeline (CLAHE 1.2 + Bilateral Denoising + High-Pass Sharpening) saved to disk (`dataset-Road-Deepglobe-Preprocessed`).
- **FocusMIM Data Augmentation (`MaskAug`)**: Dynamic road corridor $16\times16$ patch masking ($p=0.5, K=85, \text{mask\_ratio}=0.30, \text{token}=128$).
- **Resumable Training**: Automatic checkpoint detection & loading (`best_dlinknet_v5.pth`) across Kaggle input directories and local paths.

### ⚡ Version 4.0 — FocusFormer (`Road_FocusFormer_v4.ipynb`)
- **Architecture**: FocusFormer with Windowed Focus Self-Attention (W-FSA) and Channel-Spatial Attention (CSA) modules.
- **Goal**: Addresses long-range linear road continuity and occlusions caused by tree canopy and building shadows.

### 🎓 Version 3.0 — Teacher-Student Framework (`Road_Sentinel2_TeacherStudent_v3.ipynb`)
- **Architecture**: Cross-modal / Cross-resolution Teacher-Student Knowledge Distillation.
- **Goal**: Transfers rich contextual feature representations from high-resolution satellite imagery to lighter student segmenters.

---

## 🧪 3-Stage Mild Preprocessing Suite

To resolve sensor noise, atmospheric haze, and low contrast without introducing over-saturation or harsh color clipping, all advanced versions utilize our **Mild Preprocessing Suite**:

1. **Gentle CLAHE (`clipLimit=1.2`, tile grid `8x8`)**: Softly boosts shadow contrast while preserving cloud highlights.
2. **Subtle Bilateral Filter (`d=5`, `sigmaColor=25`, `sigmaSpace=25`)**: Smooths sensor grain while maintaining crisp, unblurred road edges.
3. **Mild High-Pass Sharpening**: Accentuates subtle dirt tracks and narrow asphalt road boundaries.

---

## ⚡ Quick Start & Execution

### 1. Training V5.0 D-LinkNet via Command Line
```powershell
python train_dlinknet_v5.py
```

### 2. Running in Jupyter / Kaggle / Colab
Open any of the Jupyter Notebook files:
- [**`Road_DLinkNet_v5.ipynb`**](file:///Road_DLinkNet_v5.ipynb)
- [**`Road_FocusFormer_v4.ipynb`**](file:///Road_FocusFormer_v4.ipynb)
- [**`Road_Sentinel2_TeacherStudent_v3.ipynb`**](file:///Road_Sentinel2_TeacherStudent_v3.ipynb)

Set runtime accelerator to **GPU (T4 / P100 / A100)** and execute all cells.

---
*Created for DeepGlobe Satellite Road Extraction Benchmarking.*
