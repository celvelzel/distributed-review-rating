"""Stage 0 baseline: TF-IDF features + LightGBM regressor."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def extract_tfidf_features(
    train_texts: pd.Series,
    test_texts: pd.Series,
    max_features: int = 5000,
) -> Tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    """Fit TF-IDF on *train_texts* and transform both splits.

    Parameters
    ----------
    train_texts : pd.Series
        Concatenated title + comment for training rows.
    test_texts : pd.Series
        Concatenated title + comment for test rows.
    max_features : int
        Vocabulary size cap.

    Returns
    -------
    X_train : np.ndarray  (sparse)
    X_test  : np.ndarray  (sparse)
    vectorizer : fitted TfidfVectorizer
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    X_train = vectorizer.fit_transform(train_texts.fillna(""))
    X_test = vectorizer.transform(test_texts.fillna(""))
    return X_train, X_test, vectorizer


def train_lgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Optional[dict] = None,
) -> lgb.LGBMRegressor:
    """Train a LightGBM regressor with sensible defaults.

    Parameters
    ----------
    X_train : sparse or dense array
    y_train : target ratings
    params  : optional overrides for LGBMRegressor kwargs

    Returns
    -------
    fitted LGBMRegressor
    """
    defaults = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "n_estimators": 500,
        "verbose": -1,
    }
    if params:
        defaults.update(params)

    model = lgb.LGBMRegressor(**defaults)
    model.fit(X_train, y_train)
    return model


def predict_and_save(
    model: lgb.LGBMRegressor,
    X_test: np.ndarray,
    test_ids: np.ndarray,
    output_path: str,
) -> pd.DataFrame:
    """Generate predictions, clip to [1, 5], and write submission CSV.

    Parameters
    ----------
    model       : fitted regressor
    X_test      : feature matrix for test set
    test_ids    : review id column
    output_path : where to write ``id,rating`` CSV

    Returns
    -------
    DataFrame with columns ``id`` and ``rating``.
    """
    preds = model.predict(X_test)
    preds = np.clip(preds, 1.0, 5.0)

    submission = pd.DataFrame({"id": test_ids, "rating": preds})
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    return submission
