import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif

from pipeline_modules.cleaning import load_and_preprocess_data


def plot_heatmap_and_mi():
    df = load_and_preprocess_data()

    df_encoded = df.copy()
    cat_cols = df_encoded.select_dtypes(include="object").columns.tolist()

    le = LabelEncoder()
    for c in cat_cols:
        df_encoded[c] = le.fit_transform(df_encoded[c].astype(str))

    # ---------------------------
    # Spearman Heatmap
    # ---------------------------
    spearman_corr = df_encoded.corr(method="spearman")

    plt.figure(figsize=(14, 10))
    mask = np.triu(np.ones_like(spearman_corr, dtype=bool))

    sns.heatmap(
        spearman_corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        annot_kws={"size": 7},
    )

    plt.title("Spearman Rank-Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    # ---------------------------
    # Mutual Information
    # ---------------------------
    X = df_encoded.drop(columns=["Need_Maintenance"])
    y = df_encoded["Need_Maintenance"]

    mi_scores = mutual_info_classif(
        X, y, discrete_features="auto", random_state=42
    )

    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    mi_series.plot(kind="bar")
    plt.title("Mutual Information Scores")
    plt.ylabel("MI Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_heatmap_and_mi()