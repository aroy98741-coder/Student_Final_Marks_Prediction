"""Exploratory data analysis. Run:  python src/eda.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import FEATURES, FIG_DIR, RESULTS_DIR, TARGET
from preprocessing import load_data

sns.set_theme(style="whitegrid")


def save(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=130)
    plt.close()


def main():
    df = load_data()
    cols = FEATURES + [TARGET]

    print("Shape:", df.shape)
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:", int(df.isna().sum().sum()))
    print("Duplicates:", int(df.duplicated().sum()))

    stats = df[cols].describe().T.round(2)
    print("\nDescriptive statistics:\n", stats)
    stats.to_csv(RESULTS_DIR / "descriptive_statistics.csv")

    corr = df[cols].corr()
    corr.to_csv(RESULTS_DIR / "correlation_matrix.csv")
    target_corr = corr[TARGET].drop(TARGET).sort_values(ascending=False)
    print("\nCorrelation with target:\n", target_corr.round(3))

    print("\nOutliers (IQR rule):")
    for c in cols:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        n = int(((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr)).sum())
        print(f"  {c:18s}: {n}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df[TARGET], kde=True, color="steelblue", ax=ax[0])
    ax[0].set_title("Distribution of final marks")
    sns.boxplot(x=df[TARGET], color="lightblue", ax=ax[1])
    ax[1].set_title("Boxplot of final marks")
    save("target_distribution.png")

    for c in FEATURES:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[c], kde=True, color="teal")
        plt.title(f"Distribution of {c}")
        save(f"{c}_distribution.png")

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation heatmap")
    save("correlation_heatmap.png")

    plt.figure(figsize=(7, 4))
    target_corr.sort_values().plot.barh(color="seagreen")
    plt.xlabel("Correlation with final marks")
    plt.title("Top correlations")
    save("top_correlations.png")

    n = len(FEATURES)
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, c in zip(axes.ravel(), FEATURES):
        sns.regplot(x=df[c], y=df[TARGET], ax=ax, scatter_kws=dict(alpha=0.25, s=10),
                    line_kws=dict(color="red"))
        ax.set_title(f"{c} vs final marks")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    save("scatter_relationships.png")

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, c in zip(axes.ravel(), FEATURES):
        sns.boxplot(y=df[c], ax=ax, color="khaki")
        ax.set_title(c)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    save("feature_boxplots.png")

    print(f"\nEDA figures saved in {FIG_DIR}")


if __name__ == "__main__":
    main()
