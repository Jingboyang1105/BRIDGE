"""TEEBoT-style accessible-tissue baseline matched to the HYFA notebook."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LinearRegression


def build_accessible_design(
    expression,
    age=None,
    sex=None,
    *,
    include_demographics: bool = True,
) -> np.ndarray:
    values = np.asarray(expression, dtype=np.float32)
    if values.ndim != 3 or values.shape[1] != 4:
        raise ValueError("accessible expression must contain exactly four ordered source tissues")
    if not np.isfinite(values).all():
        raise ValueError("accessible expression contains non-finite values")
    design = values.reshape(values.shape[0], -1)
    if not include_demographics:
        return design
    if age is None or sex is None:
        raise ValueError("age and sex are required when demographics are enabled")
    age_values = np.asarray(age, dtype=np.float32).reshape(-1)
    sex_values = np.asarray(sex, dtype=np.float32).reshape(-1)
    if len(age_values) != len(design) or len(sex_values) != len(design):
        raise ValueError("demographic rows must match expression rows")
    if not np.isfinite(age_values).all() or not np.isfinite(sex_values).all():
        raise ValueError("demographic inputs contain non-finite values")
    return np.concatenate(
        [design, (age_values / 100.0)[:, None], sex_values[:, None]], axis=1
    ).astype(np.float32, copy=False)


def fit_predict_teebot_style(
    x_train,
    y_train,
    x_test,
    *,
    n_components: int = 30,
    seed: int = 42,
) -> np.ndarray:
    train = np.asarray(x_train, dtype=np.float32)
    target = np.asarray(y_train, dtype=np.float32)
    test = np.asarray(x_test, dtype=np.float32)
    if train.ndim != 2 or target.ndim != 2 or test.ndim != 2:
        raise ValueError("TEEBoT-style arrays must be two-dimensional")
    if train.shape[0] < 2:
        raise ValueError("TEEBoT-style fitting requires at least two training donors")
    if target.shape[0] != train.shape[0] or test.shape[1] != train.shape[1]:
        raise ValueError("TEEBoT-style train, target and test shapes are inconsistent")
    if not all(np.isfinite(array).all() for array in (train, target, test)):
        raise ValueError("TEEBoT-style inputs contain non-finite values")
    components = min(int(n_components), train.shape[0] - 1, train.shape[1] - 1)
    if components < 1:
        raise ValueError("TEEBoT-style fitting has no valid SVD components")
    center = train.mean(axis=0, dtype=np.float64).astype(np.float32)
    svd = TruncatedSVD(n_components=components, random_state=int(seed))
    low_train = svd.fit_transform(train - center)
    low_test = svd.transform(test - center)
    prediction = LinearRegression().fit(low_train, target).predict(low_test)
    prediction = np.asarray(prediction, dtype=np.float32)
    if prediction.shape != (test.shape[0], target.shape[1]) or not np.isfinite(prediction).all():
        raise ValueError("TEEBoT-style prediction is invalid")
    return prediction
