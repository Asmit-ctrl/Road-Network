# 🛰️ Road Intelligence: DeepGlobe Satellite Road Extraction Benchmark Suite

Comprehensive benchmarking, model architectures, and experimental outcomes for satellite road segmentation on the **DeepGlobe Road Extraction Dataset**. This repository documents the training dynamics, feature enhancement pipelines, dynamic patch masking, quantitative metrics, and model architectures from **V1 Baseline** up to **V5.0 D-LinkNet34**.

---

## 🏆 Notebook Results Comparison (from saved notebook outputs)

| Notebook | Dataset | Model / Version | Val IoU | Val Dice | Test IoU | Test Dice | Notes |
|:---|:---|:---|:---:|:---:|:---:|:---:|:---|
| [`Road_BaseLine.ipynb`](file:///Road_BaseLine.ipynb) | DeepGlobe | V1 Baseline (DeepLabV3+) | `0.6637` | `0.7970` | `—` | `—` | Validation summary table is saved in output. |
| [`Road_Segmentation_FocusMIM.ipynb`](file:///Road_Segmentation_FocusMIM.ipynb) | DeepGlobe | FocusMIM + DeepLabV3+ | `0.6479` | `0.7853` | `—` | `—` | Validation summary table is saved in output. |
| [`Road_FocusFormer_v4.ipynb`](file:///Road_FocusFormer_v4.ipynb) | DeepGlobe | V4 FocusFormer (W-FSA + CSA) | `0.6850` | `0.8105` | `0.6810` | `0.8105` | Metrics are referenced in `road-intelligence-v5-0.ipynb` leaderboard output. |
| [`road-intelligence-v5-0.ipynb`](file:///road-intelligence-v5-0.ipynb) | DeepGlobe | V5 D-LinkNet34 + FocusMIM | `0.6542` | `0.7900` | `0.6594` | `0.7938` | Final test evaluation cell contains full leaderboard. |
| [`Sentinal-2_RGB_BaseLine.ipynb`](file:///Sentinal-2_RGB_BaseLine.ipynb) | Sentinel-2 RGB | Baseline DeepLabV3+ | `0.2785` (best), `0.2703` (final) | `—` | `—` | `—` | Notebook verdict: weak but non-zero signal. |
| [`Sentianl_2_RGB_Knowledge_Distillation.ipynb`](file:///Sentianl_2_RGB_Knowledge_Distillation.ipynb) | Sentinel-2 RGB | V3 Distilled Student | `—` | `—` | `0.3189` | `0.4815` | Final test block includes comparison against DeepGlobe baselines. |

---

## ✅ Conclusions from all notebooks

1. **Best DeepGlobe score in saved outputs** is the **V4 FocusFormer** result reported in the V5 notebook leaderboard (`Test IoU 0.6810`, `Test Dice 0.8105`).
2. **V1 DeepLabV3+ baseline is strong and stable** (`Val IoU 0.6637`, `Val Dice 0.7970`) and remains competitive with later variants.
3. **FocusMIM with DeepLabV3+ did not beat V1 baseline** in its standalone run (`Val IoU 0.6479`), but it is reused in the V5 D-LinkNet pipeline.
4. **V5 D-LinkNet34 + FocusMIM is robust** and produced `Test IoU 0.6594`, `Test Dice 0.7938` in the saved run.
5. **Sentinel-2 is clearly harder than DeepGlobe**: baseline is low (`Best Val IoU 0.2785`), and distillation improves test performance (`Test IoU 0.3189`, `Dice 0.4815`) but still trails DeepGlobe metrics.

---

## 📂 Repository Notebook Files

| Notebook File | Description |
|:---|:---|
| [**`road-intelligence-v5-0.ipynb`**](file:///road-intelligence-v5-0.ipynb) | **Version 5.0**: Official PyTorch D-LinkNet34 + Preprocessed Dataset + FocusMIM (`MaskAug`) + Resumable Checkpoints. |
| [**`Road_FocusFormer_v4.ipynb`**](file:///Road_FocusFormer_v4.ipynb) | **Version 4.0**: FocusFormer Model with Windowed Focus Self-Attention (W-FSA) & Channel Spatial Attention (CSA). |
| [**`Sentianl_2_RGB_Knowledge_Distillation.ipynb`**](file:///Sentianl_2_RGB_Knowledge_Distillation.ipynb) | **Version 3.0**: Teacher-Student Knowledge Distillation Framework for Satellite Road Extraction. |
| [`Road_BaseLine.ipynb`](file:///Road_BaseLine.ipynb) | **Version 1.0**: DeepGlobe baseline training and evaluation (DeepLabV3+). |
| [`Road_Segmentation_FocusMIM.ipynb`](file:///Road_Segmentation_FocusMIM.ipynb) | **Version 2.0**: DeepGlobe segmentation with FocusMIM augmentation. |
| [`Sentinal-2_RGB_BaseLine.ipynb`](file:///Sentinal-2_RGB_BaseLine.ipynb) | Sentinel-2 RGB baseline experiment and data-quality verdict run. |

---

## 🔬 Model Architectures & Methodologies

### 🚀 Version 5.0 — Official D-LinkNet34 + FocusMIM (`road-intelligence-v5-0.ipynb`)
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

### 🎓 Version 3.0 — Teacher-Student Framework (`Sentianl_2_RGB_Knowledge_Distillation.ipynb`)
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

### 1. Running in Jupyter / Kaggle / Colab
Open any of the Jupyter Notebook files:
- [**`road-intelligence-v5-0.ipynb`**](file:///road-intelligence-v5-0.ipynb)
- [**`Road_FocusFormer_v4.ipynb`**](file:///Road_FocusFormer_v4.ipynb)
- [**`Sentianl_2_RGB_Knowledge_Distillation.ipynb`**](file:///Sentianl_2_RGB_Knowledge_Distillation.ipynb)

Set runtime accelerator to **GPU (T4 / P100 / A100)** and execute all cells.

---
*Created for DeepGlobe Satellite Road Extraction Benchmarking.*
