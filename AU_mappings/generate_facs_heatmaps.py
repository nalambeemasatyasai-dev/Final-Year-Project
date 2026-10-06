# -*- coding: utf-8 -*-
"""
Upgraded Action Unit (AU) Mapping for Student Engagement Detection.
Computes:
1. Conditional Probability Mapping P(AU | Engagement) [Heatmap + Line Plot]
2. Relative Activation Ratios (RAR) [Heatmap]
3. Statistical Discriminant Coefficient (SDC) Scores [Heatmap + Bar Chart]
Uses Google MediaPipe 52 FACS Blendshapes across engagement levels:
  - Class 0: Disengaged
  - Class 1: Nominally Engaged
  - Class 2: Highly Engaged
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RESULTS_DIR = os.path.join(BASE_DIR, "Results", "AU_features")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Key representative FACS blendshapes for clear visualization
KEY_FACS_BLENDSHAPES = [
    'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight',
    'eyeBlinkLeft', 'eyeBlinkRight', 'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft',
    'eyeLookDownLeft', 'eyeLookUpLeft', 'jawOpen', 'mouthSmileLeft', 'mouthSmileRight',
    'mouthFrownLeft', 'mouthFrownRight', 'mouthPucker', 'mouthClose'
]

CLASS_LABELS = {0: "Disengaged (0)", 1: "Partially Engaged (1)", 2: "Highly Engaged (2)"}


class AU_mapping:
    def __init__(self, threshold=0.10):
        self.threshold = threshold

    def _get_columns(self, df, columns=None):
        if columns is not None:
            return [c for c in columns if c in df.columns]
        available_key = [c for c in KEY_FACS_BLENDSHAPES if c in df.columns]
        if len(available_key) >= 8:
            return available_key
        return [c for c in df.columns if c not in ['image_id', 'filename', 'class_label', 'image_path']]

    def prob_au(self, df, label, columns):
        """
        Calculates conditional probability P(AU >= threshold | Engagement = label).
        """
        subset = df[df['class_label'] == label]
        total_in_class = len(subset)
        if total_in_class == 0:
            return {c: 0.0 for c in columns}

        dic = {}
        for c in columns:
            active_count = (subset[c] >= self.threshold).sum()
            dic[c] = round(float(active_count / total_in_class), 4)
        return dic

    def conditional_probability_heatmap(self, df, columns=None, output_dir=RESULTS_DIR):
        """
        1. Computes conditional probability P(AU | Label) and generates heatmap + line plot.
        """
        columns = self._get_columns(df, columns)
        p0 = self.prob_au(df, 0, columns)
        p1 = self.prob_au(df, 1, columns)
        p2 = self.prob_au(df, 2, columns)

        df_cond = pd.DataFrame([p0, p1, p2], index=[CLASS_LABELS[0], CLASS_LABELS[1], CLASS_LABELS[2]]).T

        # Heatmap
        fig_hm, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df_cond, cmap='magma_r', linewidths=0.7, annot=True, vmin=0, vmax=1,
                    annot_kws={"size": 11}, fmt='.2f', ax=ax)
        plt.title("Conditional Probability P(FACS AU | Engagement State)", fontsize=13, pad=15, fontweight='bold')
        plt.xlabel("Engagement Level", fontsize=12)
        plt.ylabel("Facial Muscle Action Units (FACS)", fontsize=12)
        plt.tight_layout()

        png_hm = os.path.join(output_dir, "CondProb_AU_mapping.png")
        pdf_hm = os.path.join(output_dir, "CondProb_AU_mapping.pdf")
        csv_hm = os.path.join(output_dir, "CondProb_AU_mapping.csv")

        fig_hm.savefig(png_hm, dpi=150, bbox_inches='tight')
        fig_hm.savefig(pdf_hm, bbox_inches='tight')
        df_cond.to_csv(csv_hm)
        plt.close(fig_hm)

        # Line Plot
        fig_lp = plt.figure(figsize=(12, 6))
        for index, row in df_cond.iterrows():
            plt.plot(row.index, row.values, marker='o', linewidth=1.8, label=index)
        plt.title('Conditional Probability Mapping of FACS Facial Action Units Across Engagement Levels',
                  fontsize=13, fontweight='bold', pad=12)
        plt.xlabel('Engagement Level', fontsize=12)
        plt.ylabel('Activation Probability P(AU >= 0.10 | Level)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize=9, ncol=1)
        plt.tight_layout()

        png_lp = os.path.join(output_dir, "CondProb_AU_mappingLinePlot.png")
        pdf_lp = os.path.join(output_dir, "CondProb_AU_mappingLinePlot.pdf")
        fig_lp.savefig(png_lp, dpi=150, bbox_inches='tight')
        fig_lp.savefig(pdf_lp, bbox_inches='tight')
        plt.close(fig_lp)

        return df_cond, png_hm, png_lp

    def relative_activation_ratio_heatmap(self, df, columns=None, output_dir=RESULTS_DIR):
        """
        2. Computes Relative Activation Ratio (RAR):
           Ratio = P(AU | Label) / P(AU)
        """
        columns = self._get_columns(df, columns)
        total_samples = len(df)
        p_overall = {c: max((df[c] >= self.threshold).sum() / total_samples, 1e-6) for c in columns}

        ratios = {}
        for c in columns:
            ratios[c] = {}
            for label in [0, 1, 2]:
                subset = df[df['class_label'] == label]
                if len(subset) == 0:
                    ratios[c][CLASS_LABELS[label]] = 1.0
                    continue
                p_c_l = (subset[c] >= self.threshold).sum() / len(subset)
                ratios[c][CLASS_LABELS[label]] = round(p_c_l / p_overall[c], 2)

        df_rar = pd.DataFrame(ratios).T

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df_rar, cmap='magma_r', linewidths=0.7, annot=True,
                    annot_kws={"size": 11}, fmt='.2f', ax=ax)
        plt.title("Relative Activation Ratios (RAR) Across Engagement Levels\n[Ratio = P(AU | State) / P(AU)]",
                  fontsize=13, pad=15, fontweight='bold')
        plt.xlabel("Engagement Level", fontsize=12)
        plt.ylabel("Facial Muscle Action Units (FACS)", fontsize=12)
        plt.tight_layout()

        png_path = os.path.join(output_dir, "Relative_AU_mapping.png")
        pdf_path = os.path.join(output_dir, "Relative_AU_mapping.pdf")
        csv_path = os.path.join(output_dir, "Relative_AU_mapping.csv")

        fig.savefig(png_path, dpi=150, bbox_inches='tight')
        fig.savefig(pdf_path, bbox_inches='tight')
        df_rar.to_csv(csv_path)
        plt.close(fig)

        return df_rar, png_path

    def sdc_scores_analysis(self, df, columns=None, output_dir=RESULTS_DIR):
        """
        3. Computes Statistical Discriminant Coefficient (SDC) Scores:
           SDC(c) = sum_{l} P(l) * ln( P(c | l) / P(c) )
        """
        columns = self._get_columns(df, columns)
        total_samples = len(df)
        labels = [0, 1, 2]

        P_c = {c: max((df[c] >= self.threshold).sum() / total_samples, 1e-6) for c in columns}
        P_l = {l: (df['class_label'] == l).sum() / total_samples for l in labels}

        sdc_scores = {}
        for c in columns:
            sdc = 0.0
            for l in labels:
                subset = df[df['class_label'] == l]
                if len(subset) == 0:
                    continue
                p_c_l = (subset[c] >= self.threshold).sum() / len(subset)
                ratio = max(p_c_l, 1e-6) / P_c[c]
                sdc += P_l[l] * np.log(ratio)
            sdc_scores[c] = round(sdc, 4)

        df_sdc = pd.DataFrame.from_dict(
            sdc_scores, orient='index', columns=["SDC Score"]
        ).sort_values(by="SDC Score", ascending=False)

        # Heatmap
        fig_hm, ax = plt.subplots(figsize=(6, 8))
        sns.heatmap(df_sdc, cmap='magma', linewidths=0.7, annot=True,
                    annot_kws={"size": 11}, fmt='.4f', ax=ax)
        plt.title("Statistical Discriminant Coefficient\n(SDC Scores for FACS)", fontsize=13, pad=15, fontweight='bold')
        plt.ylabel("Action Units (FACS)", fontsize=12)
        plt.tight_layout()

        png_hm = os.path.join(output_dir, "SDC_Scores.png")
        pdf_hm = os.path.join(output_dir, "SDC_Scores.pdf")
        csv_path = os.path.join(output_dir, "SDC_Scores.csv")

        fig_hm.savefig(png_hm, dpi=150, bbox_inches='tight')
        fig_hm.savefig(pdf_hm, bbox_inches='tight')
        df_sdc.to_csv(csv_path)
        plt.close(fig_hm)

        # Bar chart of top discriminative blendshapes
        fig_bar, ax = plt.subplots(figsize=(10, 6))
        colors = ['#10b981' if x >= 0 else '#ef4444' for x in df_sdc["SDC Score"]]
        df_sdc["SDC Score"].plot(kind='barh', color=colors, ax=ax, edgecolor='black', linewidth=0.5)
        ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        plt.title("FACS Muscle Action Discriminative Power (SDC Scores)", fontsize=13, fontweight='bold', pad=12)
        plt.xlabel("SDC Score (Log-Ratio Discriminability)", fontsize=12)
        plt.ylabel("Facial Action Units", fontsize=12)
        plt.tight_layout()

        png_bar = os.path.join(output_dir, "SDC_Discriminative_BarPlot.png")
        pdf_bar = os.path.join(output_dir, "SDC_Discriminative_BarPlot.pdf")
        fig_bar.savefig(png_bar, dpi=150, bbox_inches='tight')
        fig_bar.savefig(pdf_bar, bbox_inches='tight')
        plt.close(fig_bar)

        return df_sdc, png_hm, png_bar


def run_all_au_mappings(df=None, output_dir=RESULTS_DIR):
    """
    Convenience function to run all 3 AU mappings (CondProb, RAR, SDC) and save all plots and CSVs.
    """
    if df is None:
        data_path = os.path.join(BASE_DIR, "WACV data", "mediapipe_blendshapes.pkl")
        if not os.path.exists(data_path):
            data_path = os.path.join(BASE_DIR, "WACV data", "mediapipe_blendshapes.csv")
        print(f"Loading blendshapes from: {data_path}")
        df = pd.read_pickle(data_path) if data_path.endswith('.pkl') else pd.read_csv(data_path)

    os.makedirs(output_dir, exist_ok=True)
    mapper = AU_mapping(threshold=0.10)

    print("  [1/3] Generating Conditional Probability Heatmap & Line Plot...")
    df_cond, png_cp_hm, png_cp_lp = mapper.conditional_probability_heatmap(df, output_dir=output_dir)

    print("  [2/3] Generating Relative Activation Ratio (RAR) Heatmap...")
    df_rar, png_rar = mapper.relative_activation_ratio_heatmap(df, output_dir=output_dir)

    print("  [3/3] Generating Statistical Discriminant Coefficient (SDC) Analysis...")
    df_sdc, png_sdc_hm, png_sdc_bar = mapper.sdc_scores_analysis(df, output_dir=output_dir)

    print(f"  AU Mapping visualizations saved to: {output_dir}")
    return {
        'cond_prob': df_cond,
        'rar': df_rar,
        'sdc': df_sdc,
        'files': [png_cp_hm, png_cp_lp, png_rar, png_sdc_hm, png_sdc_bar]
    }


def main():
    print("=" * 60)
    print("  UPGRADED FACS ACTION UNIT (AU) MAPPING ANALYSIS")
    print("=" * 60)
    results = run_all_au_mappings()
    print("\nTop 5 Most Discriminative FACS Action Units (by SDC):")
    print(results['sdc'].head(5))
    print("\nExecution complete.")


if __name__ == "__main__":
    main()