"""Predict final marks for a student.

Examples:
    python src/predict.py
    python src/predict.py --attendance 90 --test1 30 --test2 35 --assignment 8 --hours 3
"""
import argparse

import joblib
import pandas as pd

from config import FEATURES, MODEL_DIR


def load_model(name="best_model"):
    path = MODEL_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python src/train_models.py")
    return joblib.load(path)


def predict_marks(attendance, internal_test_1, internal_test_2, assignment_score,
                  study_hours, model=None):
    model = model or load_model()
    row = pd.DataFrame([[attendance, internal_test_1, internal_test_2,
                         assignment_score, study_hours]], columns=FEATURES)
    return float(min(100.0, max(0.0, model.predict(row)[0])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--attendance", type=float, default=85)
    ap.add_argument("--test1", type=float, default=32)
    ap.add_argument("--test2", type=float, default=32)
    ap.add_argument("--assignment", type=float, default=7.5)
    ap.add_argument("--hours", type=float, default=3)
    a = ap.parse_args()
    score = predict_marks(a.attendance, a.test1, a.test2, a.assignment, a.hours)
    print(f"Student: attendance={a.attendance}%, test1={a.test1}/40, test2={a.test2}/40, "
          f"assignment={a.assignment}/10, study hours={a.hours}/day")
    print(f"Predicted final exam marks: {score:.1f} / 100")
