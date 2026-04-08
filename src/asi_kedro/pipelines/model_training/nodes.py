import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

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

    category_mappings: dict[str, dict[str, int]] = {}
    for col in cat_columns:
        train_values = X_train[col].astype(str)
        unique_values = sorted(train_values.unique().tolist())
        mapping = {value: idx for idx, value in enumerate(unique_values)}
        category_mappings[col] = mapping

        # Unknown categories map to -1 to keep inference robust.
        X_train[col] = train_values.map(mapping).fillna(-1).astype(int)
        X_val[col] = X_val[col].astype(str).map(mapping).fillna(-1).astype(int)

    clf = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        random_state=params["random_state"],
        n_jobs=-1,
    )

    clf.fit(X_train, y_train)
    clf.category_mappings_ = category_mappings

    val_score = clf.score(X_val, y_val)
    logger.info("Accuracy na zbiorze walidacyjnym: %.4f", val_score)

    return clf