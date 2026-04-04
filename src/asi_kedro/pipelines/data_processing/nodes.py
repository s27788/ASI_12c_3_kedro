
import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)



# -- LIDKA

def preprocess_data(
    data: pd.DataFrame,
    parameters: Dict[str, Any],
) -> pd.DataFrame:
    """Preprocess the airline passenger satisfaction dataset."""

    df = data.copy()

    rows_before = len(df)
    df = df.drop_duplicates()
    logger.info("Removed %d duplicate rows.", rows_before - len(df))

    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
        logger.info("Dropped ID column.")

    if "Satisfaction" in df.columns:
        original_target = df["Satisfaction"].copy()
        df["Satisfaction"] = df["Satisfaction"].astype(str).str.strip().str.lower().map(
            {
                "satisfied": 1,
                "neutral or dissatisfied": 0,
            }
        )

        if df["Satisfaction"].isna().any():
            unexpected_values = original_target[df["Satisfaction"].isna()].unique().tolist()
            raise ValueError(
                f"Unexpected values found in 'Satisfaction' column after mapping: {unexpected_values}"
            )

        logger.info("Converted target 'Satisfaction' to binary.")

    cols_to_remove = [
        "Departure and Arrival Time Convenience",
        "Ease of Online Booking",
        "Check-in Service",
        "Online Boarding",
        "Gate Location",
        "On-board Service",
        "Seat Comfort",
        "Leg Room Service",
        "Cleanliness",
        "Food and Drink",
        "In-flight Service",
        "In-flight Wifi Service",
        "In-flight Entertainment",
        "Baggage Handling",
    ]

    existing_cols = [col for col in cols_to_remove if col in df.columns]
    df = df.drop(columns=existing_cols)
    logger.info("Removed %d post-flight columns.", len(existing_cols))

    if "Arrival Delay" in df.columns:
        missing_before = df["Arrival Delay"].isna().sum()
        median_value = df["Arrival Delay"].median()
        df["Arrival Delay"] = df["Arrival Delay"].fillna(median_value)
        logger.info(
            "Filled %d missing values in 'Arrival Delay'.",
            missing_before,
        )

    if "Arrival Delay" in df.columns:
        df["arrival_delay_log"] = np.log1p(df["Arrival Delay"])

    if "Departure Delay" in df.columns:
        df["departure_delay_log"] = np.log1p(df["Departure Delay"])

    delay_cols = ["Arrival Delay", "Departure Delay"]
    existing_delay_cols = [col for col in delay_cols if col in df.columns]
    df = df.drop(columns=existing_delay_cols)

    logger.info("Preprocessing completed. Shape: %s", df.shape)

    return df



# -- MARYSIA

def split_data(
    data: pd.DataFrame,
    parameters: Dict[str, Any],
) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series, pd.Series, pd.Series
]:
    """Split data into train, validation and test sets."""

    target = parameters["target_column"]
    split_params = parameters["split"]

    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found.")

    X = data.drop(columns=[target])
    y = data[target]

    # train vs temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=split_params["test_size"],
        random_state=split_params["random_state"],
        stratify=y,
    )

    # validation vs test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=split_params["val_ratio"],
        random_state=split_params["random_state"],
        stratify=y_temp,
    )

    logger.info(
        "Split done: train=%d, val=%d, test=%d",
        len(X_train), len(X_val), len(X_test)
    )

    return X_train, X_val, X_test, y_train, y_val, y_test