# -*- coding: utf-8 -*-
"""
Comprehensive Benchmark of All ML and DL Models for Student Engagement Recognition.
Compares:
1. Baseline Classical ML (OpenFace + Facial Landmarks)
2. Advanced GBDT (MediaPipe 52 FACS Blendshapes: XGBoost, LightGBM, CatBoost)
3. Baseline DL Models (ResNet-18, EfficientNet-B0 with CLAHE & 3-sample sampler)
4. Upgraded Multimodal DL (ResNet-18, ResNet-50, EfficientNet-B0, EfficientNet-B1 + FACS)
5. Vision Ablations (Image-Only ResNet-18, Image-Only EfficientNet-B0)
6. Multimodal Calibrated Ensemble (6 Models x 6-View TTA)
"""

import os
import sys
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "Results")
os.makedirs(RESULTS_DIR, exist_ok=True)

from run_master_pipeline import build_final_benchmark_tables, generate_overall_benchmark_plot


def run_benchmark():
    build_final_benchmark_tables(RESULTS_DIR)

    csv_path = os.path.join(RESULTS_DIR, "all_models_benchmark.csv")
    output_png = os.path.join(RESULTS_DIR, "overall_benchmark_comparison.png")
    output_pdf = os.path.join(RESULTS_DIR, "overall_benchmark_comparison.pdf")
    generate_overall_benchmark_plot(csv_path, output_png, output_pdf)

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print("\n" + "=" * 110)
        print(f"{'Category':<32} | {'Model Name':<38} | {'Accuracy':<10} | {'F1-Score'}")
        print("=" * 110)
        for _, r in df.iterrows():
            f1_str = r.get('F1-Score', 'N/A')
            print(f"{str(r['Category']):<32} | {str(r['Model Name']):<38} | {str(r['Accuracy']):<10} | {f1_str}")
        print("=" * 110 + "\n")
        return df
    return None


if __name__ == "__main__":
    run_benchmark()
