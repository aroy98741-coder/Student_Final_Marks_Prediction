"""Loading, cleaning and preprocessing of the student dataset."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (DATA_PATH, FEATURES, ID_COLUMNS, RANDOM_STATE, RENAME,
                    TARGET, TEST_SIZE)


def load_data(path=DATA_PATH):
    """Read the CSV, drop ID column, rename columns, remove duplicates and rows without target."""
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in ID_COLUMNS if c in df.columns])
    df = df.rename(columns=RENAME)
    df = df.drop_duplicates().dropna(subset=[TARGET]).reset_index(drop=True)
    return df


def build_preprocessor():
    """Median-impute missing values, then standardise (fitted on training data only)."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    return ColumnTransformer([("num", numeric, FEATURES)])


def split_data(df):
    X, y = df[FEATURES], df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
