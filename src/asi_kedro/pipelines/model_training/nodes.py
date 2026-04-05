import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    parameters: dict,
) -> RandomForestClassifier:
    """Trenuje RandomForestClassifier na danych treningowych.

    Args:
        X_train: Cechy zbioru treningowego.
        y_train: Target zbioru treningowego.
        X_val: Cechy zbioru walidacyjnego.
        y_val: Target zbioru walidacyjnego.
        parameters: Slownik hiperparametrow z parameters.yml.

    Returns:
        Wytrenowany model RandomForestClassifier.
    """
    params = parameters["model"]

    # Enkodowanie kolumn kategorycznych
    X_train = X_train.copy()
    X_val = X_val.copy()

    cat_columns = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    logger.info("Enkodowanie kolumn kategorycznych: %s", cat_columns)

    for col in cat_columns:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_val[col] = le.transform(X_val[col].astype(str))

    clf = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        random_state=params["random_state"],
        n_jobs=-1,
    )

    clf.fit(X_train, y_train)

    val_score = clf.score(X_val, y_val)
    logger.info("Accuracy na zbiorze walidacyjnym: %.4f", val_score)

    return clf