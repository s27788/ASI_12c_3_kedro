import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def preprocess_data(
    data: pd.DataFrame,
    parameters: Dict[str, Any],
) -> pd.DataFrame:
    """Preprocess the airline passenger satisfaction dataset.

    This function prepares the dataset for the pre-flight prediction scenario.
    It removes duplicates, drops the identifier column, converts the target to
    binary form, removes post-flight columns that may cause data leakage,
    fills missing values in arrival delay, and creates log-transformed delay
    features.

    Args:
        data: Raw dataset loaded from SQLite by Kedro.
        parameters: Project parameters loaded from Kedro configuration.

    Returns:
        Cleaned dataframe ready for the next pipeline steps.

    Raises:
        ValueError: If unexpected values are found in the target column after mapping.
    """
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
    logger.info("Removed %d post-flight columns (data leakage).", len(existing_cols))

    if "Arrival Delay" in df.columns:
        missing_before = df["Arrival Delay"].isna().sum()
        median_value = df["Arrival Delay"].median()
        df["Arrival Delay"] = df["Arrival Delay"].fillna(median_value)
        logger.info(
            "Filled %d missing values in 'Arrival Delay' with median %.2f.",
            missing_before,
            median_value,
        )

    if "Arrival Delay" in df.columns:
        df["arrival_delay_log"] = np.log1p(df["Arrival Delay"])

    if "Departure Delay" in df.columns:
        df["departure_delay_log"] = np.log1p(df["Departure Delay"])

    delay_cols = ["Arrival Delay", "Departure Delay"]
    existing_delay_cols = [col for col in delay_cols if col in df.columns]
    df = df.drop(columns=existing_delay_cols)
    logger.info("Replaced delay columns with log-transformed versions.")

    logger.info("Preprocessing completed. Output shape: %s", df.shape)

    return df