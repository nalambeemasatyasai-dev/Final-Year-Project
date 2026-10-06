# -*- coding: utf-8 -*-
"""
Upgraded Machine Learning Classification Pipeline for Student Engagement Recognition.
Uses Google MediaPipe Face Blendshapes (52 FACS muscle action scores).
Evaluates 4 Feature Subsets:
  - S0: Eye & Gaze FACS Blendshapes (Attentiveness & Focus)
  - S1: Eyebrow & Forehead FACS Blendshapes (Cognitive Load & Confusion)
  - S2: Mouth, Jaw & Cheek FACS Blendshapes (Fatigue, Affect & Engagement)
  - S3: All 52 FACS Muscle Action Units Combined
Classifiers:
  - Random Forest
  - XGBoost
  - CatBoost
  - LightGBM
  - Decision Tree
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, classification_report
)
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# S0: Eye & Gaze FACS features
S0_EYE_GAZE = [
    'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight',
    'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight',
    'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight',
    'eyeWideLeft', 'eyeWideRight'
]

# S1: Eyebrow & Forehead FACS features
S1_BROW_FOREHEAD = [
    'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight'
]

# S2: Mouth, Jaw & Cheek FACS features
S2_MOUTH_JAW_CHEEK = [
    'jawForward', 'jawLeft', 'jawOpen', 'jawRight', 'mouthClose',
    'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight',
    'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight',
    'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight',
    'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper',
    'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight',
    'mouthUpperUpLeft', 'mouthUpperUpRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight'
]

# S3: All 52 FACS Blendshapes
S3_ALL_52_FACS = [
    '_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft',
    'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight',
    'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight',
    'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight',
    'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight',
    'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawOpen', 'jawRight',
    'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft',
    'mouthFrownRight', 'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft',
    'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight', 'mouthPucker',
    'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower',
    'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft',
    'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft',
    'noseSneerRight'
]


class ML_classification:
    def stratified_split(self, df, label_col='class_label', val_ratio=0.10, test_ratio=0.10, seed=42):
        """
        Pure NumPy Stratified train/val/test splitting for reproducible data partitioning.
        """
        rng = np.random.default_rng(seed)
        train_idxs, val_idxs, test_idxs = [], [], []
        for c in sorted(df[label_col].unique()):
            c_idxs = df[df[label_col] == c].index.to_numpy()
            rng.shuffle(c_idxs)
            n = len(c_idxs)
            n_val = int(round(n * val_ratio))
            n_test = int(round(n * test_ratio))
            val_idxs.extend(c_idxs[:n_val])
            test_idxs.extend(c_idxs[n_val:n_val + n_test])
            train_idxs.extend(c_idxs[n_val + n_test:])

        return train_idxs, val_idxs, test_idxs

    def feature_sets(self, df):
        """
        Constructs train/test datasets for the 4 functional feature sets:
        - S0: Eye & Gaze FACS (14 features)
        - S1: Brow & Forehead FACS (5 features)
        - S2: Mouth, Jaw & Cheek FACS (30 features)
        - S3: All 52 FACS Blendshapes Combined (52 features)
        """
        train_idxs, val_idxs, test_idxs = self.stratified_split(df)
        train_df = df.loc[train_idxs].reset_index(drop=True)
        test_df = df.loc[test_idxs].reset_index(drop=True)

        feature_subsets = [
            ("s0", "Eye_Gaze", S0_EYE_GAZE),
            ("s1", "Brow_Forehead", S1_BROW_FOREHEAD),
            ("s2", "Mouth_Jaw_Cheek", S2_MOUTH_JAW_CHEEK),
            ("s3", "Combined_All_52_FACS", S3_ALL_52_FACS)
        ]

        data = []
        for code, name, cols in feature_subsets:
            valid_cols = [c for c in cols if c in df.columns]
            X_train = train_df[valid_cols].values.astype(np.float32)
            y_train = train_df['class_label'].values.astype(int)
            X_test = test_df[valid_cols].values.astype(np.float32)
            y_test = test_df['class_label'].values.astype(int)

            data.append({
                'code': code,
                'name': name,
                'columns': valid_cols,
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test
            })

        return data

    def classifier_result(self, dataset, feature_set_idx, output_dir=None):
        """
        Trains 5 ML classifiers on the given feature set:
        - Random Forest
        - XGBoost
        - CatBoost
        - LightGBM
        - Decision Tree
        Returns results DataFrame, ROC Curve Figure, and Model Comparison Bar Plot Figure.
        """
        X_train, y_train = dataset['X_train'], dataset['y_train']
        X_test, y_test = dataset['X_test'], dataset['y_test']

        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42),
            "XGBoost": xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42),
            "CatBoost": CatBoostClassifier(iterations=300, learning_rate=0.05, depth=6, verbose=0, random_seed=42),
            "LightGBM": lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1),
            "Decision Tree": DecisionTreeClassifier(max_depth=12, random_state=42),
            "SVM": SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
        }

        results = []
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

        # Figure 1: ROC Curves (TP vs FP) matching exact Riju format
        fig_roc = plt.figure(figsize=(6, 6))
        palette = {
            'Random Forest': '#3b82f6',
            'Gradient Boosting': '#0284c7',
            'XGBoost': '#ef4444',
            'CatBoost': '#10b981',
            'LightGBM': '#f59e0b',
            'Decision Tree': '#8b5cf6',
            'SVM': '#ec4899'
        }

        for name, clf in models.items():
            clf.fit(X_train, y_train)

            if hasattr(clf, "predict_proba"):
                probs = clf.predict_proba(X_test)
                preds = np.argmax(probs, axis=1)
            else:
                preds = clf.predict(X_test)
                probs = None

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, average='macro', zero_division=0)
            rec = recall_score(y_test, preds, average='macro', zero_division=0)
            f1 = f1_score(y_test, preds, average='macro', zero_division=0)

            # Micro ROC AUC
            roc_auc_val = 0.5
            if probs is not None:
                try:
                    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), probs.ravel())
                    roc_auc_val = auc(fpr, tpr)
                    plt.plot(fpr, tpr, lw=2, color=palette.get(name, 'gray'),
                             label=f"{name}(area = {roc_auc_val:.2f})")
                except Exception:
                    pass

            results.append({
                'Feature_Set': dataset['code'].upper(),
                'model': name,
                'Accuracy': round(float(acc), 2),
                'Precision': round(float(prec), 2),
                'Recall': round(float(rec), 2),
                'F-measure': round(float(f1), 2)
            })

        plt.plot([0, 1], [0, 1], 'k--', lw=1.2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title(f'Receiver operating characteristic: {dataset["code"].upper()} ({dataset["name"]})',
                  fontsize=12, fontweight='bold', pad=10)
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        # Figure 2: Model Comparison Bar Plot for this feature set
        df_res = pd.DataFrame(results)
        fig_bar, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(df_res))
        width = 0.35
        ax.bar(x - width/2, [a * 100 for a in df_res['Accuracy']], width, label='Accuracy (%)', color='#3b82f6', edgecolor='black', linewidth=0.5)
        ax.bar(x + width/2, [f * 100 for f in df_res['F-measure']], width, label='F1-Score (%)', color='#10b981', edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Percentage (%)', fontsize=11)
        ax.set_title(f'Model Performance on Feature Set {dataset["code"].upper()} ({dataset["name"]})',
                     fontsize=12, fontweight='bold', pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(df_res['model'], rotation=15, ha='right', fontsize=10)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 100)
        for i in x:
            ax.annotate(f"{df_res['Accuracy'].iloc[i]*100:.1f}%", (i - width/2, df_res['Accuracy'].iloc[i]*100 + 1.5),
                        ha='center', fontsize=9, fontweight='bold')
            ax.annotate(f"{df_res['F-measure'].iloc[i]*100:.1f}%", (i + width/2, df_res['F-measure'].iloc[i]*100 + 1.5),
                        ha='center', fontsize=9)
        plt.tight_layout()

        return df_res, fig_roc, fig_bar


def plot_all_feature_sets_summary(all_results_df, output_path=None):
    """
    Generates a master side-by-side grouped bar chart comparing all 4 feature sets (S0, S1, S2, S3)
    across all models.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_df = all_results_df.copy()
    plot_df['Accuracy_Pct'] = plot_df['Accuracy'] * 100
    sns.barplot(data=plot_df, x='Feature_Set', y='Accuracy_Pct', hue='model', palette='tab10', ax=ax, edgecolor='black', linewidth=0.5)
    ax.set_title('Comparative Classification Accuracy Across Feature Sets S0, S1, S2, S3',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Feature Set (S0: Eye/Gaze, S1: Brow, S2: Mouth/Jaw, S3: All 52 FACS)', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(title='Classifier', fontsize=10, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()

    if output_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        pdf_path = os.path.splitext(output_path)[0] + ".pdf"
        fig.savefig(pdf_path, bbox_inches='tight')
    plt.close(fig)
    return fig
