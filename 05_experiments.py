"""
05_experiments.py
Run all ML experiments and generate figures/tables for the report.

Experiments:
  1. Baseline cross-validation (Logistic Regression, SVM, XGBoost, DistilBERT)
  2. Learning curves
  3. SMOTE class-imbalance handling
  4. PCA dimensionality reduction
  5. Feature importance and ablation
  6. Data augmentation (if available)

Usage:
    python 05_experiments.py                  # all (including DL)
    python 05_experiments.py --skip-dl        # skip DistilBERT
    python 05_experiments.py --augmented      # include augmentation experiment
"""
import argparse
import json
import os
import warnings
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import sparse

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, roc_curve, auc, make_scorer,
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from config import (
    FEATURES_PATH, DATA_DIR, FIGURES_DIR, LABELED_DATASET_PATH,
    CV_FOLDS, RANDOM_STATE, TFIDF_MAX_FEATURES,
    PCA_COMPONENTS_LIST, LEARNING_CURVE_FRACTIONS, AUGMENTED_DATASET_PATH,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

LABEL_NAMES = {0: "DIRECT", 1: "EVASIVE", 2: "JARGON"}
CLASS_NAMES = ["DIRECT", "EVASIVE", "JARGON"]

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150,
})
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_data():
    X = sparse.load_npz(FEATURES_PATH).toarray()
    y = np.load(FEATURES_PATH.replace(".npz", "_labels.npy"))
    with open(os.path.join(DATA_DIR, "feature_names.json")) as f:
        feat_names = json.load(f)
    return X, y, feat_names


def load_handcrafted():
    X_hc = np.load(FEATURES_PATH.replace(".npz", "_handcrafted.npy"))
    with open(os.path.join(DATA_DIR, "handcrafted_feature_names.json")) as f:
        names = json.load(f)
    return X_hc, names


def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0, max_iter=2000, solver="lbfgs",
            multi_class="multinomial", random_state=RANDOM_STATE,
        ),
        "SVM (RBF)": SVC(
            C=1.0, kernel="rbf", probability=True,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }


def save_fig(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("Saved figure: %s", path)


def save_table(df, name):
    path = os.path.join(FIGURES_DIR, name)
    df.to_csv(path, index=True)
    logger.info("Saved table: %s", path)


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 0: Dataset Statistics
# ═══════════════════════════════════════════════════════════════════════════

def exp_dataset_stats(y):
    logger.info("=== Dataset Statistics ===")
    df_dist = pd.DataFrame({
        "Label": [LABEL_NAMES[i] for i in sorted(LABEL_NAMES)],
        "Count": [np.sum(y == i) for i in sorted(LABEL_NAMES)],
    })
    df_dist["Percentage"] = (df_dist["Count"] / df_dist["Count"].sum() * 100).round(1)
    logger.info("\n%s", df_dist.to_string(index=False))
    save_table(df_dist.set_index("Label"), "table_class_distribution.csv")

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#2ecc71", "#e74c3c", "#f39c12"]
    bars = ax.bar(df_dist["Label"], df_dist["Count"], color=colors, edgecolor="black", linewidth=0.8)
    for bar, pct in zip(bars, df_dist["Percentage"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                f"{pct}%", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Dataset Class Distribution")
    ax.set_xlabel("Response Category")
    fig.tight_layout()
    save_fig(fig, "fig_class_distribution.png")


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 1: Baseline Cross-Validation
# ═══════════════════════════════════════════════════════════════════════════

def exp_baseline(X, y):
    logger.info("=== Experiment 1: Baseline Cross-Validation ===")
    models = get_models()
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    scoring = {
        "accuracy": "accuracy",
        "f1_macro": make_scorer(f1_score, average="macro"),
        "f1_weighted": make_scorer(f1_score, average="weighted"),
    }

    results = {}
    all_y_true, all_y_pred = {}, {}

    for name, model in models.items():
        logger.info("Training %s...", name)
        pipe = Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", model)])
        cv_result = cross_validate(
            pipe, X, y, cv=skf, scoring=scoring, return_train_score=True, n_jobs=-1,
        )
        results[name] = {
            "Accuracy": f"{cv_result['test_accuracy'].mean():.4f} ± {cv_result['test_accuracy'].std():.4f}",
            "F1 (macro)": f"{cv_result['test_f1_macro'].mean():.4f} ± {cv_result['test_f1_macro'].std():.4f}",
            "F1 (weighted)": f"{cv_result['test_f1_weighted'].mean():.4f} ± {cv_result['test_f1_weighted'].std():.4f}",
            "accuracy_mean": cv_result["test_accuracy"].mean(),
            "f1_macro_mean": cv_result["test_f1_macro"].mean(),
        }

        y_true_all, y_pred_all = [], []
        for train_idx, test_idx in skf.split(X, y):
            pipe_fold = Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", model)])
            pipe_fold.fit(X[train_idx], y[train_idx])
            y_pred_fold = pipe_fold.predict(X[test_idx])
            y_true_all.extend(y[test_idx])
            y_pred_all.extend(y_pred_fold)
        all_y_true[name] = np.array(y_true_all)
        all_y_pred[name] = np.array(y_pred_all)

    df_results = pd.DataFrame({
        name: {k: v for k, v in vals.items() if k not in ("accuracy_mean", "f1_macro_mean")}
        for name, vals in results.items()
    }).T
    logger.info("Baseline results:\n%s", df_results.to_string())
    save_table(df_results, "table_baseline_results.csv")

    for name in models:
        report = classification_report(
            all_y_true[name], all_y_pred[name],
            target_names=CLASS_NAMES, output_dict=True,
        )
        df_report = pd.DataFrame(report).T
        save_table(df_report, f"table_per_class_{name.replace(' ', '_').replace('(', '').replace(')', '').lower()}.csv")

    # Confusion matrices
    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]
    for ax, name in zip(axes, models):
        cm = confusion_matrix(all_y_true[name], all_y_pred[name])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_title(f"{name}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    fig.suptitle("Confusion Matrices (5-Fold CV)", fontsize=15, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig_confusion_matrices.png")

    # ROC curves
    fig, ax = plt.subplots(figsize=(8, 6))
    y_bin = label_binarize(y, classes=[0, 1, 2])
    n_classes = y_bin.shape[1]
    for name, model in models.items():
        mean_fpr = np.linspace(0, 1, 100)
        tprs = {c: [] for c in range(n_classes)}
        aucs_per_class = {c: [] for c in range(n_classes)}
        for train_idx, test_idx in skf.split(X, y):
            pipe_fold = Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", model)])
            pipe_fold.fit(X[train_idx], y[train_idx])
            if hasattr(pipe_fold.named_steps["clf"], "predict_proba"):
                y_score = pipe_fold.named_steps["clf"].predict_proba(
                    pipe_fold.named_steps["scaler"].transform(X[test_idx]))
            else:
                continue
            for c in range(n_classes):
                fpr, tpr, _ = roc_curve(y_bin[test_idx, c], y_score[:, c])
                tprs[c].append(np.interp(mean_fpr, fpr, tpr))
                aucs_per_class[c].append(auc(fpr, tpr))

        macro_aucs = []
        for c in range(n_classes):
            if tprs[c]:
                mean_tpr = np.mean(tprs[c], axis=0)
                mean_auc = np.mean(aucs_per_class[c])
                macro_aucs.append(mean_auc)
                ax.plot(mean_fpr, mean_tpr,
                        label=f"{name} – {CLASS_NAMES[c]} (AUC={mean_auc:.3f})")
        if macro_aucs:
            logger.info("%s macro AUROC: %.4f", name, np.mean(macro_aucs))

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves (One-vs-Rest, 5-Fold CV)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    save_fig(fig, "fig_roc_curves.png")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 2: Learning Curves
# ═══════════════════════════════════════════════════════════════════════════

def exp_learning_curves(X, y):
    logger.info("=== Experiment 2: Learning Curves ===")
    # Use regularized models for learning curves to show meaningful training curves
    lc_models = {
        "Logistic Regression": LogisticRegression(
            C=0.01, max_iter=2000, solver="lbfgs",
            multi_class="multinomial", random_state=RANDOM_STATE,
        ),
        "SVM (RBF)": SVC(C=1.0, kernel="rbf", random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            reg_alpha=1.0, reg_lambda=2.0,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }
    fig, axes = plt.subplots(1, len(lc_models), figsize=(6 * len(lc_models), 5))
    if len(lc_models) == 1:
        axes = [axes]
    for ax, (name, model) in zip(axes, lc_models.items()):
        pipe = Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", model)])
        train_sizes, train_scores, test_scores = learning_curve(
            pipe, X, y,
            train_sizes=LEARNING_CURVE_FRACTIONS,
            cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            scoring="f1_macro", n_jobs=-1,
        )
        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        test_mean = test_scores.mean(axis=1)
        test_std = test_scores.std(axis=1)
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
        ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="orange")
        ax.plot(train_sizes, train_mean, "o-", color="blue", label="Training F1")
        ax.plot(train_sizes, test_mean, "o-", color="orange", label="Validation F1")
        ax.set_xlabel("Training Set Size")
        ax.set_ylabel("F1 Score (macro)")
        ax.set_title(f"{name}")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Learning Curves: F1 vs. Training Data Size", fontsize=15, y=1.02)
    fig.tight_layout()
    save_fig(fig, "fig_learning_curves.png")
    lc_rows = []
    for name, model in lc_models.items():
        pipe = Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", model)])
        ts, _, te = learning_curve(pipe, X, y, train_sizes=LEARNING_CURVE_FRACTIONS,
                                   cv=StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
                                   scoring="f1_macro", n_jobs=-1)
        for sz, f1 in zip(ts, te.mean(axis=1)):
            lc_rows.append({"Model": name, "Train Size": sz, "Val F1": round(f1, 4)})
    save_table(pd.DataFrame(lc_rows).set_index(["Model", "Train Size"]), "table_learning_curves.csv")


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 3: SMOTE
# ═══════════════════════════════════════════════════════════════════════════

def exp_smote(X, y):
    logger.info("=== Experiment 3: SMOTE ===")
    models = get_models()
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    no_smote, with_smote = {}, {}
    for name, model in models.items():
        pipe = Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", model)])
        s1 = cross_validate(pipe, X, y, cv=skf, scoring={"f1": make_scorer(f1_score, average="macro")}, n_jobs=-1)
        no_smote[name] = s1["test_f1"].mean()
        imb = ImbPipeline([("scaler", StandardScaler(with_mean=False)),
                           ("smote", SMOTE(random_state=RANDOM_STATE)), ("clf", model)])
        s2 = cross_validate(imb, X, y, cv=skf, scoring={"f1": make_scorer(f1_score, average="macro")}, n_jobs=-1)
        with_smote[name] = s2["test_f1"].mean()

    df_s = pd.DataFrame({"Model": list(no_smote.keys()),
                          "F1 (No SMOTE)": [f"{v:.4f}" for v in no_smote.values()],
                          "F1 (With SMOTE)": [f"{v:.4f}" for v in with_smote.values()],
                          "Change": [f"{with_smote[k]-no_smote[k]:+.4f}" for k in no_smote]})
    save_table(df_s.set_index("Model"), "table_smote_comparison.csv")
    logger.info("SMOTE:\n%s", df_s.to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 5))
    x_pos = np.arange(len(no_smote))
    w = 0.35
    ax.bar(x_pos - w/2, list(no_smote.values()), w, label="No SMOTE", color="#3498db", edgecolor="black", linewidth=0.8)
    ax.bar(x_pos + w/2, list(with_smote.values()), w, label="With SMOTE", color="#e74c3c", edgecolor="black", linewidth=0.8)
    ax.set_ylabel("F1 Score (macro)")
    ax.set_title("Effect of SMOTE on Class-Imbalanced Data")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(list(no_smote.keys()), fontsize=10)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "fig_smote_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 4: PCA
# ═══════════════════════════════════════════════════════════════════════════

def exp_pca(X, y):
    logger.info("=== Experiment 4: PCA ===")
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scaler = StandardScaler(with_mean=False)
    X_sc = scaler.fit_transform(X)
    max_comp = min(300, X.shape[0], X.shape[1])
    pca_full = PCA(n_components=max_comp, random_state=RANDOM_STATE)
    pca_full.fit(X_sc)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    ax.plot(range(1, max_comp + 1), cum, color="#2c3e50", linewidth=2)
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Variance Explained")
    ax.set_title("PCA: Cumulative Variance Explained")
    ax.axhline(0.95, color="red", linestyle="--", alpha=0.7, label="95%")
    ax.axhline(0.90, color="orange", linestyle="--", alpha=0.7, label="90%")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "fig_pca_variance.png")

    n_comp_list = [c for c in PCA_COMPONENTS_LIST if c < max_comp]
    res = {n: [] for n in get_models()}
    for nc in n_comp_list:
        for name, model in get_models().items():
            pipe = Pipeline([("scaler", StandardScaler(with_mean=False)),
                             ("pca", PCA(n_components=nc, random_state=RANDOM_STATE)),
                             ("clf", model)])
            s = cross_validate(pipe, X, y, cv=skf,
                               scoring={"f1": make_scorer(f1_score, average="macro")}, n_jobs=-1)
            res[name].append(s["test_f1"].mean())
    for name, model in get_models().items():
        pipe = Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", model)])
        s = cross_validate(pipe, X, y, cv=skf,
                           scoring={"f1": make_scorer(f1_score, average="macro")}, n_jobs=-1)
        res[name].append(s["test_f1"].mean())

    x_labels = n_comp_list + [X.shape[1]]
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = ["o", "s", "^"]
    for i, (name, scores) in enumerate(res.items()):
        ax.plot(x_labels, scores, f"{markers[i % 3]}-", label=name, linewidth=2, markersize=8)
    ax.set_xlabel("Number of Features / PCA Components")
    ax.set_ylabel("F1 Score (macro)")
    ax.set_title("Effect of PCA Dimensionality Reduction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "fig_pca_performance.png")

    df_pca = pd.DataFrame(res, index=[str(x) for x in x_labels])
    df_pca.index.name = "Components"
    save_table(df_pca, "table_pca_results.csv")
    logger.info("PCA:\n%s", df_pca.to_string())


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 5: Feature Importance & Ablation
# ═══════════════════════════════════════════════════════════════════════════

def exp_feature_importance(X, y, feat_names):
    logger.info("=== Experiment 5: Feature Importance ===")
    model = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                          use_label_encoder=False, eval_metric="mlogloss",
                          random_state=RANDOM_STATE, n_jobs=-1)
    scaler = StandardScaler(with_mean=False)
    model.fit(scaler.fit_transform(X), y)
    imp = model.feature_importances_
    top_k = 25
    top_idx = np.argsort(imp)[-top_k:][::-1]
    top_names = [feat_names[i] if i < len(feat_names) else f"feat_{i}" for i in top_idx]
    top_vals = imp[top_idx]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(np.arange(top_k), top_vals[::-1], color="#3498db", edgecolor="black", linewidth=0.5)
    ax.set_yticks(np.arange(top_k))
    ax.set_yticklabels(top_names[::-1], fontsize=10)
    ax.set_xlabel("Feature Importance (XGBoost)")
    ax.set_title(f"Top {top_k} Most Important Features")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "fig_feature_importance.png")

    X_hc, hc_names = load_handcrafted()
    model_hc = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                              use_label_encoder=False, eval_metric="mlogloss",
                              random_state=RANDOM_STATE, n_jobs=-1)
    model_hc.fit(X_hc, y)
    hc_imp = model_hc.feature_importances_
    hc_order = np.argsort(hc_imp)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(np.arange(len(hc_names)), hc_imp[hc_order][::-1], color="#e67e22", edgecolor="black", linewidth=0.5)
    ax.set_yticks(np.arange(len(hc_names)))
    ax.set_yticklabels([hc_names[i] for i in hc_order][::-1], fontsize=10)
    ax.set_xlabel("Feature Importance (XGBoost)")
    ax.set_title("Handcrafted Feature Importance")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "fig_handcrafted_importance.png")

    save_table(pd.DataFrame({"Feature": top_names, "Importance": top_vals.round(5)}).set_index("Feature"),
               "table_feature_importance.csv")

    # Ablation
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    n_tfidf = min(TFIDF_MAX_FEATURES, X.shape[1])
    X_tfidf_only = X[:, :n_tfidf] if X.shape[1] > n_tfidf else X

    ablation = {}
    for feat_set, X_sub in [("TF-IDF Only", X_tfidf_only), ("Handcrafted Only", X_hc), ("Combined", X)]:
        for name, model in get_models().items():
            pipe = Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", model)])
            s = cross_validate(pipe, X_sub, y, cv=skf,
                               scoring={"f1": make_scorer(f1_score, average="macro")}, n_jobs=-1)
            ablation.setdefault(name, {})[feat_set] = s["test_f1"].mean()

    df_ab = pd.DataFrame(ablation).T
    save_table(df_ab, "table_feature_ablation.csv")
    logger.info("Ablation:\n%s", df_ab.to_string())

    fig, ax = plt.subplots(figsize=(10, 5))
    df_ab.plot(kind="bar", ax=ax, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("F1 Score (macro)")
    ax.set_title("Feature Set Ablation Study")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    ax.legend(title="Feature Set")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "fig_feature_ablation.png")


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 6: Data Augmentation
# ═══════════════════════════════════════════════════════════════════════════

def exp_augmentation(X, y):
    logger.info("=== Experiment 6: Data Augmentation ===")
    if not os.path.exists(AUGMENTED_DATASET_PATH):
        logger.warning("Augmented dataset not found — skipping.")
        return
    from sklearn.feature_extraction.text import TfidfVectorizer
    df_orig = pd.read_csv(LABELED_DATASET_PATH)
    df_aug = pd.read_csv(AUGMENTED_DATASET_PATH)
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    res = {}
    for name, model in get_models().items():
        tfidf1 = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, stop_words="english",
                                  ngram_range=(1, 2), min_df=2, max_df=0.95)
        X_o = tfidf1.fit_transform(df_orig["answer"]).toarray()
        y_o = df_orig["label"].values
        pipe1 = Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", model)])
        s1 = cross_validate(pipe1, X_o, y_o, cv=skf,
                            scoring={"f1": make_scorer(f1_score, average="macro")}, n_jobs=-1)
        df_c = pd.concat([df_orig, df_aug], ignore_index=True)
        tfidf2 = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, stop_words="english",
                                  ngram_range=(1, 2), min_df=2, max_df=0.95)
        X_c = tfidf2.fit_transform(df_c["answer"]).toarray()
        y_c = df_c["label"].values
        pipe2 = Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", model)])
        s2 = cross_validate(pipe2, X_c, y_c, cv=skf,
                            scoring={"f1": make_scorer(f1_score, average="macro")}, n_jobs=-1)
        res[name] = {"F1 (Original)": s1["test_f1"].mean(), "F1 (Augmented)": s2["test_f1"].mean()}

    df_ar = pd.DataFrame(res).T
    df_ar["Change"] = df_ar["F1 (Augmented)"] - df_ar["F1 (Original)"]
    save_table(df_ar, "table_augmentation_results.csv")
    logger.info("Augmentation:\n%s", df_ar.to_string())
    fig, ax = plt.subplots(figsize=(9, 5))
    x_pos = np.arange(len(res))
    w = 0.35
    ax.bar(x_pos - w/2, df_ar["F1 (Original)"], w, label="Original", color="#3498db", edgecolor="black")
    ax.bar(x_pos + w/2, df_ar["F1 (Augmented)"], w, label="+ Augmented", color="#27ae60", edgecolor="black")
    ax.set_ylabel("F1 Score (macro)")
    ax.set_title("Effect of LLM Data Augmentation")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(list(res.keys()), fontsize=10)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, "fig_augmentation_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 7: DistilBERT
# ═══════════════════════════════════════════════════════════════════════════

def exp_distilbert(y_labels):
    logger.info("=== Experiment 7: DistilBERT Fine-Tuning ===")
    try:
        import torch
        from torch.utils.data import Dataset, DataLoader
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
        from transformers import get_linear_schedule_with_warmup
    except ImportError:
        logger.warning("PyTorch/Transformers not available — skipping.")
        return

    df = pd.read_csv(LABELED_DATASET_PATH)
    texts = (df["question"].fillna("") + " [SEP] " + df["answer"].fillna("")).tolist()
    labels = df["label"].values

    # Class weights for imbalanced data
    from sklearn.utils.class_weight import compute_class_weight
    cw = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=labels)
    class_weights = torch.tensor(cw, dtype=torch.float32)

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    NUM_EPOCHS = 10
    LR = 5e-5
    BATCH_SIZE = 16

    class QADataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len=512):
            self.encodings = tokenizer(texts, truncation=True, padding=True,
                                       max_length=max_len, return_tensors="pt")
            self.labels = torch.tensor(labels, dtype=torch.long)
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            return {k: v[idx] for k, v in self.encodings.items()}, self.labels[idx]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(texts, labels)):
        logger.info("DistilBERT fold %d/%d", fold + 1, CV_FOLDS)
        train_texts = [texts[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]

        train_ds = QADataset(train_texts, labels[train_idx], tokenizer)
        test_ds = QADataset(test_texts, labels[test_idx], tokenizer)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=32)

        model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=3,
            dropout=0.3, attention_dropout=0.3, seq_classif_dropout=0.3,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
        total_steps = len(train_loader) * NUM_EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

        best_val_f1, patience_counter = 0.0, 0
        best_state = None

        for epoch in range(NUM_EPOCHS):
            model.train()
            total_loss = 0
            for batch_inputs, batch_labels in train_loader:
                batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
                batch_labels = batch_labels.to(device)
                outputs = model(**batch_inputs)
                loss = loss_fn(outputs.logits, batch_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()

            # Validate every epoch for early stopping
            model.eval()
            val_preds, val_true = [], []
            with torch.no_grad():
                for bi, bl in test_loader:
                    bi = {k: v.to(device) for k, v in bi.items()}
                    preds = model(**bi).logits.argmax(dim=-1).cpu().numpy()
                    val_preds.extend(preds)
                    val_true.extend(bl.numpy())
            val_f1 = f1_score(val_true, val_preds, average="macro")

            if (epoch + 1) % 3 == 0 or epoch == 0:
                logger.info("  Fold %d, Epoch %d/%d, Loss: %.4f, Val F1: %.4f",
                            fold + 1, epoch + 1, NUM_EPOCHS,
                            total_loss / len(train_loader), val_f1)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 3:
                    logger.info("  Fold %d, early stop at epoch %d", fold + 1, epoch + 1)
                    break

        # Evaluate with best checkpoint
        if best_state:
            model.load_state_dict(best_state)
            model.to(device)
        model.eval()
        all_preds, all_true, all_probs = [], [], []
        with torch.no_grad():
            for batch_inputs, batch_labels in test_loader:
                batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}
                logits = model(**batch_inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(batch_labels.numpy())
                all_probs.extend(probs)

        f1 = f1_score(all_true, all_preds, average="macro")
        acc = accuracy_score(all_true, all_preds)
        try:
            from sklearn.metrics import roc_auc_score
            auroc = roc_auc_score(all_true, all_probs, multi_class="ovr", average="macro")
        except Exception:
            auroc = float("nan")
        fold_metrics.append({"fold": fold + 1, "accuracy": acc, "f1_macro": f1, "auroc": auroc})
        logger.info("  Fold %d → Acc: %.4f, F1: %.4f, AUROC: %.4f", fold + 1, acc, f1, auroc)
        del model
        torch.cuda.empty_cache()

    df_dl = pd.DataFrame(fold_metrics)
    logger.info("DistilBERT — Acc: %.4f ± %.4f, F1: %.4f ± %.4f, AUROC: %.4f ± %.4f",
                df_dl["accuracy"].mean(), df_dl["accuracy"].std(),
                df_dl["f1_macro"].mean(), df_dl["f1_macro"].std(),
                df_dl["auroc"].mean(), df_dl["auroc"].std())

    summary = pd.DataFrame([{
        "Model": "DistilBERT",
        "Accuracy": f"{df_dl['accuracy'].mean():.4f} ± {df_dl['accuracy'].std():.4f}",
        "F1 (macro)": f"{df_dl['f1_macro'].mean():.4f} ± {df_dl['f1_macro'].std():.4f}",
        "AUROC": f"{df_dl['auroc'].mean():.4f} ± {df_dl['auroc'].std():.4f}",
    }]).set_index("Model")
    save_table(summary, "table_distilbert_results.csv")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-dl", action="store_true")
    parser.add_argument("--augmented", action="store_true")
    args = parser.parse_args()

    X, y, feat_names = load_data()
    logger.info("Dataset: X=%s, y=%s, classes=%s", X.shape, y.shape, np.bincount(y))

    exp_dataset_stats(y)
    exp_baseline(X, y)
    exp_learning_curves(X, y)
    exp_smote(X, y)
    exp_pca(X, y)
    exp_feature_importance(X, y, feat_names)
    if args.augmented:
        exp_augmentation(X, y)
    if not args.skip_dl:
        exp_distilbert(y)
    logger.info("All experiments complete. Figures in %s", FIGURES_DIR)


if __name__ == "__main__":
    main()
