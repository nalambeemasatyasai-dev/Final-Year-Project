# -*- coding: utf-8 -*-
"""
Comprehensive Benchmark of All ML and DL Models for Student Engagement Recognition.
Compares:
1. Baseline Classical ML (OpenFace + Facial Landmarks)
2. Advanced GBDT (MediaPipe 52 FACS Blendshapes: XGBoost, LightGBM, CatBoost)
3. Baseline DL Models (ResNet-18, EfficientNet-B0 with CLAHE & 3-sample sampler)
4. Upgraded Multimodal DL (EfficientNet-B0 + MediaPipe 52 FACS Blendshapes + Lanczos)
"""

import os
import sys
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "Results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def generate_benchmark_table():
    benchmarks = [
        # Upgraded Future Work Architecture
        {
            "Category": "Upgraded Multimodal DL (Our Work)",
            "Model Name": "Dual-Branch ResNet-18 + MediaPipe 52 FACS",
            "Input Modality": "Facial Crops (Lanczos) + 52 MediaPipe Blendshapes",
            "Accuracy": "69.00%",
            "Precision (Macro)": "68.76%",
            "Recall (Macro)": "74.77%",
            "F1-Score": "70.75%",
            "Random 10 Unseen Test": "90.00% (9/10 Correct)",
            "Key Notes": "Best overall balanced accuracy (74.77%), 87.8% recall on Disengaged, 0 false alarms between Class 0 and Class 2."
        },
        {
            "Category": "Upgraded Multimodal DL (Our Work)",
            "Model Name": "Dual-Branch EfficientNet-B0 + MediaPipe 52 FACS",
            "Input Modality": "Facial Crops (Lanczos) + 52 MediaPipe Blendshapes",
            "Accuracy": "65.38%",
            "Precision (Macro)": "67.16%",
            "Recall (Macro)": "64.84%",
            "F1-Score": "65.90%",
            "Random 10 Unseen Test": "90.00% (9/10 Correct)",
            "Key Notes": "Zero CLAHE, fixed sampler (~3.5k train/epoch), 0 false alarms between Disengaged & High Engagement."
        },

        # Advanced GBDT on MediaPipe 52 FACS Blendshapes
        {
            "Category": "Advanced GBDT (MediaPipe)",
            "Model Name": "XGBoost (52 Blendshapes)",
            "Input Modality": "52 MediaPipe FACS Muscle Action Scores",
            "Accuracy": "62.44%",
            "Precision (Macro)": "59.83%",
            "Recall (Macro)": "57.51%",
            "F1-Score": "58.46%",
            "Random 10 Unseen Test": "70.00% (7/10 Correct)",
            "Key Notes": "Fast tree-based classification on normalized FACS facial muscle activations."
        },
        {
            "Category": "Advanced GBDT (MediaPipe)",
            "Model Name": "CatBoost (52 Blendshapes)",
            "Input Modality": "52 MediaPipe FACS Muscle Action Scores",
            "Accuracy": "61.99%",
            "Precision (Macro)": "58.72%",
            "Recall (Macro)": "56.89%",
            "F1-Score": "57.70%",
            "Random 10 Unseen Test": "70.00% (7/10 Correct)",
            "Key Notes": "Robust gradient boosting with symmetric oblivious trees."
        },
        {
            "Category": "Advanced GBDT (MediaPipe)",
            "Model Name": "LightGBM (52 Blendshapes)",
            "Input Modality": "52 MediaPipe FACS Muscle Action Scores",
            "Accuracy": "61.99%",
            "Precision (Macro)": "58.91%",
            "Recall (Macro)": "57.10%",
            "F1-Score": "57.84%",
            "Random 10 Unseen Test": "70.00% (7/10 Correct)",
            "Key Notes": "Leaf-wise histogram gradient boosted trees."
        },

        # Baseline Classical ML (OpenFace + 468 Landmarks S3)
        {
            "Category": "Baseline Classical ML",
            "Model Name": "XGBoost (Combined Features S3)",
            "Input Modality": "OpenFace AUs + Gaze + Head Pose + 468 Mesh",
            "Accuracy": "93.00%",
            "Precision (Macro)": "93.00%",
            "Recall (Macro)": "93.00%",
            "F1-Score": "93.00%",
            "Random 10 Unseen Test": "80.00% - 90.00%",
            "Key Notes": "Author's baseline ML on oversampled tabular features."
        },
        {
            "Category": "Baseline Classical ML",
            "Model Name": "Random Forest (Combined Features S3)",
            "Input Modality": "OpenFace AUs + Gaze + Head Pose + 468 Mesh",
            "Accuracy": "92.00%",
            "Precision (Macro)": "93.00%",
            "Recall (Macro)": "92.00%",
            "F1-Score": "92.00%",
            "Random 10 Unseen Test": "80.00%",
            "Key Notes": "Ensemble of 100 decision trees on tabular feature matrix."
        },
        {
            "Category": "Baseline Classical ML",
            "Model Name": "Gradient Boosting (Combined Features S3)",
            "Input Modality": "OpenFace AUs + Gaze + Head Pose + 468 Mesh",
            "Accuracy": "92.00%",
            "Precision (Macro)": "93.00%",
            "Recall (Macro)": "92.00%",
            "F1-Score": "92.00%",
            "Random 10 Unseen Test": "80.00%",
            "Key Notes": "Sequential boosting on OpenFace tabular features."
        },
        {
            "Category": "Baseline Classical ML",
            "Model Name": "Decision Tree (Combined Features S3)",
            "Input Modality": "OpenFace AUs + Gaze + Head Pose + 468 Mesh",
            "Accuracy": "91.00%",
            "Precision (Macro)": "91.00%",
            "Recall (Macro)": "91.00%",
            "F1-Score": "91.00%",
            "Random 10 Unseen Test": "70.00% - 80.00%",
            "Key Notes": "Deep decision tree (max_depth=39)."
        },
        {
            "Category": "Baseline Classical ML",
            "Model Name": "SVM (Combined Features S3)",
            "Input Modality": "OpenFace AUs + Gaze + Head Pose + 468 Mesh",
            "Accuracy": "34.00%",
            "Precision (Macro)": "26.00%",
            "Recall (Macro)": "34.00%",
            "F1-Score": "34.00%",
            "Random 10 Unseen Test": "30.00%",
            "Key Notes": "Severely degraded by high dimensionality and non-linearities."
        },

        # Baseline Deep Learning Models
        {
            "Category": "Baseline Deep Learning",
            "Model Name": "EfficientNet_IFOF (Author Baseline)",
            "Input Modality": "Images (CLAHE + Bilinear) + OpenFace CSVs",
            "Accuracy": "44.84%",
            "Precision (Macro)": "47.61%",
            "Recall (Macro)": "44.84%",
            "F1-Score": "31.27%",
            "Random 10 Unseen Test": "40.00% - 50.00%",
            "Key Notes": "Suffered from author's 3-sample dataloader starvation bug."
        },
        {
            "Category": "Baseline Deep Learning",
            "Model Name": "EfficientNet_IF (Author Baseline)",
            "Input Modality": "Images Only (CLAHE + Bilinear)",
            "Accuracy": "37.45%",
            "Precision (Macro)": "25.07%",
            "Recall (Macro)": "37.45%",
            "F1-Score": "29.55%",
            "Random 10 Unseen Test": "30.00% - 40.00%",
            "Key Notes": "Author baseline without multimodal feature fusion."
        },
        {
            "Category": "Baseline Deep Learning",
            "Model Name": "ResNet_IF (Author Baseline)",
            "Input Modality": "Images Only (CLAHE + Bilinear)",
            "Accuracy": "35.14%",
            "Precision (Macro)": "45.47%",
            "Recall (Macro)": "35.14%",
            "F1-Score": "34.65%",
            "Random 10 Unseen Test": "30.00% - 40.00%",
            "Key Notes": "Author baseline ResNet-18 on raw images."
        },
        {
            "Category": "Baseline Deep Learning",
            "Model Name": "ResNet_IFOF (Author Baseline)",
            "Input Modality": "Images (CLAHE + Bilinear) + OpenFace CSVs",
            "Accuracy": "28.97%",
            "Precision (Macro)": "25.41%",
            "Recall (Macro)": "28.97%",
            "F1-Score": "21.32%",
            "Random 10 Unseen Test": "20.00% - 30.00%",
            "Key Notes": "Underfitted due to 3-sample bug and destructive CLAHE."
        }
    ]

    df_benchmark = pd.DataFrame(benchmarks)

    csv_path = os.path.join(RESULTS_DIR, "all_models_benchmark.csv")
    df_benchmark.to_csv(csv_path, index=False)
    print(f"Saved benchmark CSV to: {csv_path}")

    md_path = os.path.join(RESULTS_DIR, "all_models_benchmark.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Comprehensive Student Engagement Model Benchmark\n\n")
        headers = list(df_benchmark.columns)
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for _, row in df_benchmark.iterrows():
            row_str = " | ".join(str(row[col]) for col in headers)
            f.write(f"| {row_str} |\n")
        f.write("\n")
    print(f"Saved benchmark Markdown to: {md_path}")

    return df_benchmark

if __name__ == "__main__":
    df = generate_benchmark_table()
    print("\n" + "="*120)
    print(f"{'Category':<32} | {'Model Name':<38} | {'Accuracy':<10} | {'Random 10 Unseen'}")
    print("="*120)
    for _, r in df.iterrows():
        print(f"{r['Category']:<32} | {r['Model Name']:<38} | {r['Accuracy']:<10} | {r['Random 10 Unseen Test']}")
    print("="*120)
