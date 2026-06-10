from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PassengerFeatures(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [
                {
                    "Gender": "Female",
                    "Age": 35,
                    "Customer Type": "Returning",
                    "Type of Travel": "Business",
                    "Class": "Business",
                    "Flight Distance": 1200,
                    "arrival_delay_log": 0.69,
                    "departure_delay_log": 1.10,
                }
            ]
        },
    )

    Gender: Literal["Female", "Male"] = Field(
        ...,
        description="Passenger gender.",
    )
    Age: int = Field(
        ...,
        ge=1,
        le=120,
        description=(
            "Passenger age in years. Typical training range: 7–85; "
            "API accepts 1–120."
        ),
    )
    customer_type: Literal["First-time", "Returning"] = Field(
        ...,
        alias="Customer Type",
        description="Whether the passenger is a first-time or returning customer.",
    )
    type_of_travel: Literal["Business", "Personal"] = Field(
        ...,
        alias="Type of Travel",
        description="Purpose of the flight.",
    )
    Class: Literal["Business", "Economy", "Economy Plus"] = Field(
        ...,
        description="Ticket class.",
    )
    flight_distance: int = Field(
        ...,
        alias="Flight Distance",
        ge=1,
        le=10000,
        description=(
            "Flight distance in miles. Typical training range: 31–4983; "
            "API accepts 1–10000."
        ),
    )
    arrival_delay_log: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description=(
            "Arrival delay as log1p(minutes). Use 0.0 for no delay; "
            "e.g. 0.69 ≈ log1p(1 minute)."
        ),
    )
    departure_delay_log: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        description=(
            "Departure delay as log1p(minutes). Use 0.0 for no delay; "
            "e.g. 1.10 ≈ log1p(2 minutes)."
        ),
    )


class PredictionResponse(BaseModel):
    prediction: int = Field(
        ...,
        description="Predicted satisfaction: 1 = satisfied, 0 = neutral or dissatisfied.",
    )
    model_name: str = Field(
        ...,
        description="Name of the AutoGluon model used for the prediction.",
    )


class ModelInfoResponse(BaseModel):
    loaded: bool = Field(..., description="Whether a prediction model is loaded in memory.")
    model_name: str | None = Field(
        default=None,
        description="Name of the best AutoGluon model.",
    )
    model_type: str | None = Field(
        default=None,
        description="Type of the loaded predictor.",
    )
    problem_type: str | None = Field(
        default=None,
        description="ML problem type, e.g. binary classification.",
    )
    label: str | None = Field(
        default=None,
        description="Target column name used during training.",
    )
    eval_metric: str | None = Field(
        default=None,
        description="Evaluation metric used by AutoGluon during training.",
    )
    model_path: str | None = Field(
        default=None,
        description="Relative path to the saved model directory.",
    )
    feature_count: int | None = Field(
        default=None,
        description="Number of input features expected by the model.",
    )
    features: list[str] | None = Field(
        default=None,
        description="List of input feature names expected by the model.",
    )
