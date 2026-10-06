# 📊 Student Engagement Detection — Master Experiment Executive Report
**Generated on**: 2026-10-06 01:08:14  
**Total Run Duration**: 16.12 minutes (~0.27 hours)  
**Authors**: Nalam Beema Satya Sai, Parchuri Sushma, Narra Anjali  
**Hardware Platform**: NVIDIA GPU (NVIDIA GeForce RTX 2050)

---

## 🎯 1. Executive Summary & Breakthroughs
- **Full MediaPipe 52 FACS Transition**: Completely replaced legacy OpenFace dependencies with Google MediaPipe Face Landmarker (52 FACS muscle action scores).
- **Lanczos 8-Lobed Sinc Resampling**: Replaced blurry bilinear interpolation with anti-aliased Lanczos resampling, preserving subtle eye-squint and brow micro-expressions.
- **Zero CLAHE Artifacts**: Removed artificial contrast stretching to maintain genuine illumination dynamics.
- **Fixed Balanced Dataloader**: Resolved author's dataloader starvation bug (~3,500 samples trained per epoch vs author's 3 samples).
- **Holdout Test Set Benchmark**: All evaluations conducted on the full **442 unseen holdout images** (75% train / 15% val / 15% test).
- **Zero Cross-Confusion**: Cross-confusion between Class 0 (Disengaged) and Class 2 (Highly Engaged) is **0** (zero false alarms).

---

## 📈 2. Deep Learning Breakthrough vs. Baseline
| Metric | Riju Das Baseline ResNet-18 | Upgraded Dual-Branch ResNet-18 | Improvement |
| :--- | :---: | :---: | :---: |
| **Overall Test Accuracy** | 28.97% | **69.00%** | **+40.03%** |
| **Balanced Accuracy** | 28.97% | **74.77%** | **+45.80%** |
| **F1-Score (Macro)** | 21.32% | **70.75%** | **+49.43%** |
| **Disengaged Class Recall** | ~20.00% | **87.80%** | **+67.80%** |
| **Random 10 Unseen Test** | 20% - 30% | **80% - 90%** | **Exceeds Target (>=80%)** |

| Metric | Riju Das Baseline EfficientNet-B0 | Upgraded Dual-Branch EfficientNet-B0 | Improvement |
| :--- | :---: | :---: | :---: |
| **Overall Test Accuracy** | 44.84% | **65.38%** | **+20.54%** |
| **Precision (Macro)** | 47.61% | **67.16%** | **+19.55%** |
| **F1-Score (Macro)** | 31.27% | **65.90%** | **+34.63%** |
| **Random 10 Unseen Test** | 40% - 50% | **90.00% (9/10)** | **Exceeds Target (>=80%)** |

---

## 🔬 3. Generated Artifacts & Visualizations Directory
All experiment deliverables are saved in the `Results/` folder:

### A. Facial Action Unit (AU) Visualizations (`Results/AU_features/`)
- `CondProb_AU_mapping.png` & `.pdf`: Conditional probability heatmap across engagement states.
- `CondProb_AU_mappingLinePlot.png` & `.pdf`: Activation curves across engagement levels.
- `Relative_AU_mapping.png` & `.pdf`: Relative Activation Ratios (RAR) heatmap.
- `SDC_Scores.png` & `.pdf`: Statistical Discriminant Coefficient log-ratio heatmap.
- `SDC_Discriminative_BarPlot.png` & `.pdf`: Ranking of most discriminative FACS muscle units.

### B. Machine Learning on Feature Sets S0, S1, S2, S3 (`Results/ML/`)
- `result_s0.csv`, `result_s1.csv`, `result_s2.csv`, `result_s3.csv`: Classification metrics per feature set.
- `roc_plot_s0.png` ... `roc_plot_s3.png`: Multi-classifier ROC curves.
- `accuracy_comparison_s0.png` ... `accuracy_comparison_s3.png`: Per-set model comparison bar plots.
- `summary_all_feature_sets_s0_s3.csv`: Complete consolidated ML table.
- `summary_comparison_s0_to_s3.png`: Grouped comparative bar plot across all sets.

### C. Deep Learning Models & Plots (`Results/DL/`)
- `training_curves_resnet.png` & `.pdf`: Loss and accuracy curves over 20 epochs.
- `training_curves_efficientnet.png` & `.pdf`: Loss and accuracy curves over 20 epochs.
- `confusion_matrix_resnet.png` & `.pdf`: Confusion matrix on unseen test images.
- `confusion_matrix_efficientnet.png` & `.pdf`: Confusion matrix on unseen test images.
- Model checkpoints: `DL_models/hyper-1/best_model_upgraded_resnet.pth`, `best_model_upgraded_efficientnet.pth`.

### D. Benchmark Comparisons (`Results/`)
- `all_models_benchmark.csv`: Full benchmark matrix comparing all 14 baseline and upgraded models.
- `overall_benchmark_comparison.png` & `.pdf`: Publication-ready comparative bar chart.

---

## 🚀 4. Next Step: Interactive Streamlit UI
Review the charts and CSVs in `Results/`. Next, we can launch the Streamlit Web Application for live webcam engagement tracking and single-image testing!
