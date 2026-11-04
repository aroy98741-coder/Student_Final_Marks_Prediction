"""Basic tests. Run:  pytest"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import FEATURES, TARGET  # noqa: E402
from preprocessing import build_preprocessor, load_data, split_data  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402


def test_load_data_has_expected_columns():
    df = load_data()
    assert set(FEATURES + [TARGET]).issubset(df.columns)
    assert "Student_ID" not in df.columns


def test_no_missing_target_and_valid_range():
    df = load_data()
    assert df[TARGET].notna().all()
    assert df[TARGET].between(0, 100).all()


def test_split_sizes():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    assert len(X_train) + len(X_test) == len(df)
    assert abs(len(X_test) / len(df) - 0.2) < 0.01


def test_model_beats_baseline():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    pipe = Pipeline([("prep", build_preprocessor()), ("model", LinearRegression())])
    pipe.fit(X_train, y_train)
    assert r2_score(y_test, pipe.predict(X_test)) > 0.7
