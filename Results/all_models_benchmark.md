# Comprehensive Student Engagement Model Benchmark

Updated on: 2026-10-11 23:39:13

| Category | Model Name | Input Modality | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score | Key Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Upgraded Multimodal DL (Our Work) | Dual-Branch ResNet-18 + MediaPipe 52 FACS | Facial Crops (Lanczos) + 52 MediaPipe Blendshapes | 67.65% | 65.24% | 73.85% | 68.21% | Anchor architecture; balanced accuracy, 0 false alarms between Class 0 & 2. |
| Upgraded Multimodal DL (Our Work) | Dual-Branch EfficientNet-B0 + MediaPipe 52 FACS | Facial Crops (Lanczos) + 52 MediaPipe Blendshapes | 65.16% | 66.14% | 69.22% | 67.15% | Mobile inverted bottleneck + FACS geometry. |
| Upgraded Multimodal DL (Our Work) | Dual-Branch EfficientNet-B1 + MediaPipe 52 FACS | Facial Crops (Lanczos) + 52 MediaPipe Blendshapes | 65.61% | 63.89% | 72.59% | 66.87% | Compound scaled EfficientNet-B1 + FACS. |
| Upgraded Multimodal DL (Our Work) | Dual-Branch ResNet-50 + MediaPipe 52 FACS | Facial Crops (Lanczos) + 52 MediaPipe Blendshapes | 62.44% | 61.84% | 66.49% | 63.57% | 50-layer Bottleneck residual features (2,048 dims) + FACS. |
| Ablation DL (Vision Only) | Image-Only ResNet-18 | Facial Crops (224x224 Lanczos) Only | 66.74% | 66.09% | 73.62% | 67.54% | Ablation without FACS blendshapes. |
| Ablation DL (Vision Only) | Image-Only EfficientNet-B0 | Facial Crops (224x224 Lanczos) Only | 62.67% | 62.54% | 68.17% | 63.25% | Ablation without FACS blendshapes. |
| Upgraded Multimodal DL (Our Work) | Multimodal Ensemble (Calibrated + FACS) | Facial Crops (Lanczos) + 52 FACS + 6-View TTA | 69.00% | 69.74% | 70.58% | 70.14% | Full calibrated ensemble; highest generalisation. |
| Classical ML (MediaPipe 52 FACS) | XGBoost (52 Blendshapes) | 52 MediaPipe FACS Muscle Action Scores | 62.67% | 60.17% | 61.65% | 60.85% | Evaluated on 52 MediaPipe FACS under zero-leakage stratified split. |
| Classical ML (MediaPipe 52 FACS) | CatBoost (52 Blendshapes) | 52 MediaPipe FACS Muscle Action Scores | 62.44% | 58.83% | 63.50% | 60.61% | Evaluated on 52 MediaPipe FACS under zero-leakage stratified split. |
| Classical ML (MediaPipe 52 FACS) | Random Forest (52 Blendshapes) | 52 MediaPipe FACS Muscle Action Scores | 60.41% | 56.33% | 59.80% | 57.56% | Evaluated on 52 MediaPipe FACS under zero-leakage stratified split. |
| Classical ML (MediaPipe 52 FACS) | Gradient Boosting (52 Blendshapes) | 52 MediaPipe FACS Muscle Action Scores | 60.63% | 56.96% | 61.77% | 58.67% | Evaluated on 52 MediaPipe FACS under zero-leakage stratified split. |
| Classical ML (MediaPipe 52 FACS) | Decision Tree (52 Blendshapes) | 52 MediaPipe FACS Muscle Action Scores | 56.56% | 52.85% | 59.44% | 53.92% | Evaluated on 52 MediaPipe FACS under zero-leakage stratified split. |
| Classical ML (MediaPipe 52 FACS) | SVM (52 Blendshapes) | 52 MediaPipe FACS Muscle Action Scores | 64.48% | 61.61% | 60.06% | 60.77% | Evaluated on 52 MediaPipe FACS under zero-leakage stratified split. |
| Baseline Deep Learning | EfficientNet_IFOF (Author Baseline) | Images (CLAHE + Bilinear) + OpenFace CSVs | 44.84% | 47.61% | 44.84% | 31.27% | Suffered from author's 3-sample dataloader starvation bug. |
| Baseline Deep Learning | ResNet_IF (Author Baseline) | Images Only (CLAHE + Bilinear) | 35.14% | 45.47% | 35.14% | 34.65% | Author baseline ResNet-18 on raw images. |
| Baseline Deep Learning | ResNet_IFOF (Author Baseline) | Images (CLAHE + Bilinear) + OpenFace CSVs | 28.97% | 25.41% | 28.97% | 21.32% | Underfitted due to 3-sample bug and destructive CLAHE. |

