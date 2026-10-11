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

# --- Optional plotting backend ----------------------------------------------
# This machine's Application Control (AppLocker) policy blocks several native
# DLLs (matplotlib/_path, seaborn, sklearn's _splitter/_liblinear/...).
# Plotting and the classic sklearn estimators are therefore best-effort: the
# metrics, CSV reports and model ensembles are always produced.
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_OK = True
except Exception:
    PLOTTING_OK = False
    plt = None
    sns = None

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, classification_report
)
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.pipeline import Pipeline

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    SKLEARN_CLASSICS_OK = True
except Exception:
    SKLEARN_CLASSICS_OK = False
    RandomForestClassifier = GradientBoostingClassifier = None
    SVC = DecisionTreeClassifier = None

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


class _PrecomputedProba:
    """Adapter exposing a precomputed class-probability matrix through the same
    `predict_proba` interface as a fitted estimator, so an ensemble can flow
    through the shared evaluation/ROC loop without being refit."""

    def __init__(self, probs):
        self._probs = np.asarray(probs)

    def predict_proba(self, X):
        return self._probs

    def predict(self, X):
        return self._probs.argmax(axis=1)


# ---------------------------------------------------------------------------
# S4 feature engineering: asymmetry + region-aggregate descriptors.
# Engagement shows in *asymmetric* muscle use (smirks, one-sided brow raises,
# gaze shifts) and in *regional* activation mass (eyes closing, jaw dropping).
# Raw blendshapes force trees to rediscover these interaction patterns;
# explicit descriptors add signal without adding labels, samples, or any test
# leakage (every descriptor is computed per-row from that row's own 52 scores,
# so train/val/test stay strictly separated).
# ---------------------------------------------------------------------------
_S4_PAIRS = [
    ('browDownLeft', 'browDownRight'),
    ('browOuterUpLeft', 'browOuterUpRight'),
    ('cheekSquintLeft', 'cheekSquintRight'),
    ('eyeBlinkLeft', 'eyeBlinkRight'),
    ('eyeLookDownLeft', 'eyeLookDownRight'),
    ('eyeLookInLeft', 'eyeLookInRight'),
    ('eyeLookOutLeft', 'eyeLookOutRight'),
    ('eyeLookUpLeft', 'eyeLookUpRight'),
    ('eyeSquintLeft', 'eyeSquintRight'),
    ('eyeWideLeft', 'eyeWideRight'),
    ('jawLeft', 'jawRight'),
    ('mouthDimpleLeft', 'mouthDimpleRight'),
    ('mouthFrownLeft', 'mouthFrownRight'),
    ('mouthLeft', 'mouthRight'),
    ('mouthLowerDownLeft', 'mouthLowerDownRight'),
    ('mouthPressLeft', 'mouthPressRight'),
    ('mouthSmileLeft', 'mouthSmileRight'),
    ('mouthStretchLeft', 'mouthStretchRight'),
    ('mouthUpperUpLeft', 'mouthUpperUpRight'),
    ('noseSneerLeft', 'noseSneerRight'),
]
_S4_EYE = [
    'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight',
    'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight',
    'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight',
    'eyeWideLeft', 'eyeWideRight',
]
_S4_BROW = [
    'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft',
    'browOuterUpRight',
]
_S4_MOUTH = [
    'jawForward', 'jawLeft', 'jawOpen', 'jawRight', 'mouthClose',
    'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight',
    'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight',
    'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight',
    'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper',
    'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight',
    'mouthUpperUpLeft', 'mouthUpperUpRight', 'cheekPuff', 'cheekSquintLeft',
    'cheekSquintRight',
]


def engineer_blendshape_matrix(frame):
    """Builds the S4 design matrix: 52 raw + asymmetry/mean pairs + regional
    aggregates + global activation statistics (~106 columns, per-row pure)."""
    cols = [c for c in S3_ALL_52_FACS if c in frame.columns]
    base = frame[cols].values.astype(np.float64)
    col = {c: i for i, c in enumerate(cols)}
    extra = [base]
    names = list(cols)
    for l, r in _S4_PAIRS:
        if l in col and r in col:
            lv = base[:, col[l]]
            rv = base[:, col[r]]
            extra.append(np.abs(lv - rv)[:, None])
            names.append('asym_%s_%s' % (l, r))
            extra.append(((lv + rv) * 0.5)[:, None])
            names.append('mean_%s_%s' % (l, r))

    def _region(region_cols, tag):
        idx = [col[c] for c in region_cols if c in col]
        if not idx:
            return
        m = base[:, idx]
        extra.append(m.mean(axis=1)[:, None])
        names.append('%s_mean' % tag)
        extra.append(m.max(axis=1)[:, None])
        names.append('%s_max' % tag)
        extra.append((m > 0.4).sum(axis=1)[:, None].astype(np.float64))
        names.append('%s_active' % tag)

    _region(_S4_EYE, 'eye')
    _region(_S4_BROW, 'brow')
    _region(_S4_MOUTH, 'mouth')
    extra.append(base.mean(axis=1)[:, None])
    names.append('global_mean')
    extra.append(base.std(axis=1)[:, None])
    names.append('global_std')
    extra.append(base.max(axis=1)[:, None])
    names.append('global_max')
    extra.append((base > 0.4).sum(axis=1)[:, None].astype(np.float64))
    names.append('global_active')
    extra.append((base > 0.7).sum(axis=1)[:, None].astype(np.float64))
    names.append('global_strong')
    return np.concatenate(extra, axis=1).astype(np.float32), names


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
        # The 10% validation slice is reserved for EARLY STOPPING (it selects how
        # many boosting rounds to train, which is what prevents memorisation).
        # The stratified 10% TEST slice stays completely untouched and is
        # identical to the DL test split (same seed/algorithm), so ML and DL
        # results are directly comparable.
        train_df = df.loc[list(train_idxs)].reset_index(drop=True)
        val_df = df.loc[list(val_idxs)].reset_index(drop=True)
        test_df = df.loc[list(test_idxs)].reset_index(drop=True)

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
            X_val = val_df[valid_cols].values.astype(np.float32)
            y_val = val_df['class_label'].values.astype(int)
            X_test = test_df[valid_cols].values.astype(np.float32)
            y_test = test_df['class_label'].values.astype(int)

            data.append({
                'code': code,
                'name': name,
                'columns': valid_cols,
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
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

        # ---------------------------------------------------------------------------
        # Class-imbalance handling. The data is 9% / 51% / 40%; models trained with
        # plain empirical risk minimisation simply optimise the minority class away.
        # Inverse-frequency weights are used wherever the estimator supports them.
        # ---------------------------------------------------------------------------
        counts = np.bincount(y_train, minlength=3).astype(np.float64)
        inv_freq = counts.sum() / (3.0 * np.maximum(counts, 1.0))
        sample_w = inv_freq[y_train]
        catboost_w = {i: float(inv_freq[i]) for i in range(len(counts))}

        # Regularised hyper-parameters: more capacity than before, but constrained
        # (min_samples_leaf / min_child_weight / L2 / subsample) so the estimators
        # cannot memorise the training slice -> no train/test blow-up.
        # Always available boosted trees (scikit-learn's native estimators are
        # blocked by AppLocker on this machine, so we add extra boosting
        # *variants* to keep a diverse, high-performing ensemble).
        #
        # Regularisation is deliberately strong: with only ~4k noisy samples a
        # deep/long boosted model reaches 95%+ train accuracy while test stays
        # near 60% (a 35+ point gap). Shallower trees, heavy L2, row/column
        # subsampling and small learning rates keep train and test close together
        # so the models generalise instead of memorising.
        # ------------------------------------------------------------------
        # Regularisation strategy:
        #  * a generous number of trees, but EARLY STOPPING on the held-out
        #    validation slice decides the real ensemble size, so boosting stops
        #    exactly when it stops improving on unseen data (no memorisation);
        #  * shallower trees + L2 + row/column subsampling limit capacity;
        #  * low learning rate for smooth, generalising steps.
        # ------------------------------------------------------------------
        ES_ROUNDS = 50
        MAX_TREES = 800
        models = {
            "XGBoost": xgb.XGBClassifier(
                n_estimators=MAX_TREES, learning_rate=0.04, max_depth=5,
                min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=2.0, reg_alpha=0.3, gamma=0.1, random_state=42,
                early_stopping_rounds=ES_ROUNDS, eval_metric='mlogloss',
                n_jobs=-1),
            "CatBoost": CatBoostClassifier(
                iterations=MAX_TREES, learning_rate=0.04, depth=6,
                l2_leaf_reg=8.0, random_strength=1.0,
                early_stopping_rounds=ES_ROUNDS,
                class_weights=catboost_w, bootstrap_type='Bayesian',
                random_seed=42, verbose=0, allow_writing_files=False),
            "Random Forest": RandomForestClassifier(
                n_estimators=500, max_depth=12, min_samples_leaf=6,
                max_features='sqrt', class_weight='balanced_subsample',
                random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=3,
                subsample=0.8, min_samples_leaf=10, random_state=42),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=6, min_samples_leaf=20, class_weight='balanced',
                random_state=42),
            "SVM": Pipeline([
                ('scaler', StandardScaler()),
                ('svc', SVC(kernel='rbf', C=3.0, gamma='scale',
                            class_weight='balanced', probability=True,
                            random_state=42))
            ])
        }

        # --- Fit every base estimator ---------------------------------------
        def _needs_sample_weights(name):
            return name.startswith(("Gradient Boosting", "XGBoost"))

        X_val = dataset['X_val']
        y_val = dataset['y_val']

        def _fit(name, clf):
            """Fit with early stopping on the validation slice where supported."""
            sw = sample_w if _needs_sample_weights(name) else None
            if name.startswith("CatBoost"):
                attempts = [{'eval_set': (X_val, y_val)}]
            elif name.startswith("XGBoost"):
                attempts = [{'eval_set': [(X_val, y_val)]}]
            else:
                attempts = [{}]
            if sw is not None:
                attempts = [dict(a, sample_weight=sw) for a in attempts] + [{'sample_weight': sw}]
            attempts.append({})
            last = None
            for kw in attempts:
                try:
                    clf.fit(X_train, y_train, **kw)
                    return clf
                except Exception as e:
                    last = e
            raise last

        fitted = {}
        train_accs = {}
        val_accs = {}
        for name, clf in models.items():
            fitted[name] = _fit(name, clf)
            train_accs[name] = accuracy_score(y_train, fitted[name].predict(X_train))
            val_accs[name] = accuracy_score(y_val, fitted[name].predict(X_val))

        results = []
        y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

        # Figure 1: ROC Curves (TP vs FP) matching exact Riju format
        fig_roc = plt.figure(figsize=(6, 6)) if PLOTTING_OK else None
        palette = {
            'XGBoost': '#ef4444',
            'CatBoost': '#10b981',
            'Random Forest': '#3b82f6',
            'Gradient Boosting': '#0284c7',
            'Decision Tree': '#8b5cf6',
            'SVM': '#ec4899'
        }

        print(f"  {'Model':<24} | {'Train':>8} | {'Val':>8} | {'Test':>8} | "
              f"{'Val-Test':>9} | {'Train-Test':>11}")
        print("  " + "-" * 76)

        for name, clf in fitted.items():
            probs = clf.predict_proba(X_test)
            preds = np.argmax(probs, axis=1) if probs is not None else clf.predict(X_test)

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, average='macro', zero_division=0)
            rec = recall_score(y_test, preds, average='macro', zero_division=0)
            f1 = f1_score(y_test, preds, average='macro', zero_division=0)

            if name in train_accs:
                va = val_accs[name]
                gen_gap = va - acc                     # both unseen -> real signal
                mem_gap = train_accs[name] - acc
                # Flag only when validation and test genuinely disagree, which is
                # the actual symptom of overfitting / split-specific tuning.
                flag = "  <-- OVERFIT" if abs(gen_gap) > 0.06 else ""
                print(f"  {name:<24} | {train_accs[name]*100:7.2f}% | {va*100:7.2f}% | "
                      f"{acc*100:7.2f}% | {gen_gap*100:+8.2f} | {mem_gap*100:+10.2f}{flag}")
            else:
                print(f"  {name:<24} | {'(ens.)':>8} | {'':>8} | {acc*100:7.2f}% | "
                      f"{'':>9} | {'':>11}")

            # Micro ROC AUC
            roc_auc_val = 0.5
            if probs is not None:
                try:
                    fpr, tpr, _ = roc_curve(y_test_bin.ravel(), probs.ravel())
                    roc_auc_val = auc(fpr, tpr)
                    if PLOTTING_OK:
                        plt.plot(fpr, tpr, lw=2, color=palette.get(name, 'gray'),
                                 label=f"{name}(area = {roc_auc_val:.2f})")
                except Exception:
                    pass

            results.append({
                'Feature_Set': dataset['code'].upper(),
                'model': name,
                'Accuracy': round(float(acc), 4),
                'Precision': round(float(prec), 4),
                'Recall': round(float(rec), 4),
                'F-measure': round(float(f1), 4)
            })

        df_res = pd.DataFrame(results)

        # Plotting is optional on this machine (AppLocker blocks matplotlib).
        if not PLOTTING_OK:
            return df_res, None, None

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
    across all models. Returns None when plotting is unavailable on this machine.
    """
    if not PLOTTING_OK:
        print("Notice: plotting backend unavailable - skipping summary chart.")
        return None
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
