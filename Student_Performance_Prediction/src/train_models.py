"""Train and compare regression models. Run:  python src/train_models.py"""
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from config import FEATURES, FIG_DIR, MODEL_DIR, RANDOM_STATE, RESULTS_DIR
from preprocessing import build_preprocessor, load_data, split_data


def save(name):
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=130)
    plt.close()


def get_models():
    return {
        "linear_regression": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def main():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {X_train.shape}  Test: {X_test.shape}")

    rows, fitted, preds = [], {}, {}
    for name, model in get_models().items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        cv_r2 = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2").mean()
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rows.append(dict(Model=name, MAE=mean_absolute_error(y_test, pred),
                         RMSE=float(np.sqrt(mean_squared_error(y_test, pred))),
                         R2=r2_score(y_test, pred), CV_R2=cv_r2))
        fitted[name], preds[name] = pipe, pred
        joblib.dump(pipe, MODEL_DIR / f"{name}.pkl")

    results = pd.DataFrame(rows).sort_values("R2", ascending=False).round(3)
    print("\nModel comparison (test set):\n", results.to_string(index=False))
    results.to_csv(RESULTS_DIR / "model_results.csv", index=False)

    best = results.iloc[0]["Model"]
    joblib.dump(fitted[best], MODEL_DIR / "best_model.pkl")
    with open(RESULTS_DIR / "results_summary.json", "w") as f:
        json.dump({"best_model": best, "results": results.to_dict(orient="records")}, f, indent=2)
    print(f"\nBest model: {best}")

    plt.figure(figsize=(7, 4))
    sns.barplot(data=results, x="Model", y="R2", color="steelblue")
    plt.title("Model comparison (test R2)"); plt.xticks(rotation=15)
    save("model_comparison.png")

    pred = preds[best]
    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(y_test, pred, alpha=0.4, color="teal")
    lim = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    plt.plot(lim, lim, "r--", label="Perfect prediction")
    plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.legend()
    plt.title(f"Actual vs Predicted ({best})")
    save("actual_vs_predicted.png")

    plt.figure(figsize=(6, 4))
    sns.histplot(y_test - pred, kde=True, color="darkorange")
    plt.title("Residuals (Actual - Predicted)"); plt.xlabel("Error")
    save("residuals.png")

    plt.figure(figsize=(6, 4))
    plt.scatter(pred, y_test - pred, alpha=0.4, color="purple")
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted"); plt.ylabel("Residual")
    plt.title("Residuals vs Predicted")
    save("residuals_vs_predicted.png")

    for name, label in [("random_forest", "Random_Forest"), ("gradient_boosting", "Gradient_Boosting")]:
        imp = pd.Series(fitted[name].named_steps["model"].feature_importances_, index=FEATURES).sort_values()
        plt.figure(figsize=(7, 4))
        imp.plot.barh(color="seagreen")
        plt.title(f"Feature importance ({label})")
        save(f"feature_importance_{label}.png")

    perm = permutation_importance(fitted[best], X_test, y_test, n_repeats=15,
                                  random_state=RANDOM_STATE, scoring="r2")
    imp = pd.Series(perm.importances_mean, index=FEATURES).sort_values()
    print("\nPermutation importance of best model (drop in R2):\n",
          imp.sort_values(ascending=False).round(4))
    plt.figure(figsize=(7, 4))
    imp.plot.barh(xerr=pd.Series(perm.importances_std, index=FEATURES)[imp.index], color="seagreen")
    plt.xlabel("Drop in R2 when shuffled")
    plt.title(f"Permutation importance ({best})")
    save("permutation_importance_best_model.png")

    print(f"\nModels saved in {MODEL_DIR}\nFigures saved in {FIG_DIR}")


if __name__ == "__main__":
    main()
