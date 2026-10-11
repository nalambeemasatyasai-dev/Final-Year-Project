# -*- coding: utf-8 -*-
"""
Main training script for Upgraded Machine Learning models using MediaPipe 52 FACS Blendshapes.
Evaluates 4 functional feature sets:
  - S0: Eye & Gaze FACS Blendshapes (Attentiveness)
  - S1: Eyebrow & Forehead FACS Blendshapes (Cognitive Load)
  - S2: Mouth, Jaw & Cheek FACS Blendshapes (Fatigue & Affect)
  - S3: All 52 FACS Muscle Action Units Combined
Classifiers:
  - XGBoost
  - CatBoost
  - Random Forest
  - Gradient Boosting
  - Decision Tree
  - SVM
Generates CSV results, ROC curve plots, and model comparison bar charts.
"""

import os
import sys
import argparse
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(SCRIPT_DIR)

from gbdt_classifier_evaluator import ML_classification, plot_all_feature_sets_summary, PLOTTING_OK


def train_and_evaluate_ml(data_path=None, results_path=None):
    if results_path is None:
        results_path = os.path.join(BASE_DIR, "Results", "ML")
    os.makedirs(results_path, exist_ok=True)

    if data_path is None:
        data_path = os.path.join(BASE_DIR, "WACV data", "mediapipe_blendshapes.pkl")
        if not os.path.exists(data_path):
            data_path = os.path.join(BASE_DIR, "WACV data", "mediapipe_blendshapes.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"MediaPipe blendshapes file not found at {data_path}!")

    print(f"Loading MediaPipe blendshapes from: {data_path}")
    df = pd.read_pickle(data_path) if data_path.endswith('.pkl') else pd.read_csv(data_path)

    print(f"Total samples: {len(df)} | Classes: {dict(df['class_label'].value_counts())}")

    classifier = ML_classification()
    feature_datasets = classifier.feature_sets(df)

    all_results = []

    for i, dataset in enumerate(feature_datasets):
        code = dataset['code']
        name = dataset['name']
        print(f"\n" + "=" * 65)
        print(f"  [Feature Set {code.upper()}]: {name} ({len(dataset['columns'])} FACS Blendshapes)")
        print(f"  Train: {len(dataset['X_train'])} | Test: {len(dataset['X_test'])}")
        print("=" * 65)

        df_res, fig_roc, fig_bar = classifier.classifier_result(dataset, i, results_path)
        print(df_res.to_string(index=False))
        all_results.append(df_res)

        # Save files matching Riju's exact file names: oc_plot_s*.pdf/png and result_s*.csv
        res_csv = os.path.join(results_path, f"result_{code}.csv")
        oc_pdf = os.path.join(results_path, f"oc_plot_{code}.pdf")
        oc_png = os.path.join(results_path, f"oc_plot_{code}.png")
        bar_png = os.path.join(results_path, f"accuracy_comparison_{code}.png")
        bar_pdf = os.path.join(results_path, f"accuracy_comparison_{code}.pdf")

        df_res.to_csv(res_csv, index=False)
        if PLOTTING_OK and fig_roc is not None and fig_bar is not None:
            fig_roc.savefig(oc_png, dpi=150, bbox_inches='tight')
            fig_roc.savefig(oc_pdf, bbox_inches='tight')
            fig_bar.savefig(bar_png, dpi=150, bbox_inches='tight')
            fig_bar.savefig(bar_pdf, bbox_inches='tight')
            print(f"  Saved: {res_csv}, {oc_pdf}, {oc_png}")
        else:
            print(f"  Saved: {res_csv}  (plots skipped - plotting backend unavailable)")

    # Master summary across S0, S1, S2, S3
    master_df = pd.concat(all_results, ignore_index=True)
    master_csv = os.path.join(results_path, "summary_all_feature_sets_s0_s3.csv")
    master_df.to_csv(master_csv, index=False)

    summary_png = os.path.join(results_path, "summary_comparison_s0_to_s3.png")
    plot_all_feature_sets_summary(master_df, summary_png)
    print(f"\nMaster summary saved to: {master_csv} and {summary_png}")

    return master_df


def main():
    parser = argparse.ArgumentParser(description="Train ML models on MediaPipe Blendshapes")
    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    train_and_evaluate_ml(args.data_file, args.output_dir)


if __name__ == "__main__":
    main()