import os
import sqlite3
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent
DB_PATH = BASE_DIR / "data" / "01_raw" / "dataset.db"
SYNTHETIC_DATA_PATH = BASE_DIR / "data" / "03_primary" / "synthetic_data.csv"


def get_api_url() -> str:
    try:
        return st.secrets["API_URL"]
    except (FileNotFoundError, KeyError):
        return os.getenv("API_URL", "http://localhost:8000")


API_URL = get_api_url()

st.set_page_config(page_title="Airline Satisfaction & SDV System", layout="wide")
st.title("Airline Satisfaction - Panel Kontrolny")


@st.cache_data
def load_data() -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql("SELECT * FROM airline_data", conn)


@st.cache_resource
def fit_synthesizer(real_data: pd.DataFrame) -> GaussianCopulaSynthesizer:
    metadata = Metadata.detect_from_dataframe(data=real_data)
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(real_data)
    return synthesizer


tab_pred, tab_data, tab_synth = st.tabs(["Predykcja", "Dane", "Dane syntetyczne"])

with tab_pred:
    st.header("Predykcja satysfakcji pasażera")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", options=["Female", "Male"])
            age = st.number_input("Age", min_value=1, max_value=120, value=35)
            customer_type = st.selectbox("Customer Type", options=["Returning", "First-time"])

        with col2:
            type_of_travel = st.selectbox("Type of Travel", options=["Business", "Personal"])
            ticket_class = st.selectbox("Class", options=["Business", "Economy", "Economy Plus"])
            flight_distance = st.number_input("Flight Distance", min_value=1, max_value=10000, value=1200)

        with col3:
            arrival_delay_log = st.slider("Arrival Delay - log1p(minutes)", 0.0, 10.0, 0.69, 0.01)
            departure_delay_log = st.slider("Departure Delay - log1p(minutes)", 0.0, 10.0, 1.10, 0.01)

        submit_button = st.form_submit_button("Przewiduj satysfakcję")

    if submit_button:
        payload = {
            "Gender": gender,
            "Age": int(age),
            "Customer Type": customer_type,
            "Type of Travel": type_of_travel,
            "Class": ticket_class,
            "Flight Distance": int(flight_distance),
            "arrival_delay_log": float(arrival_delay_log),
            "departure_delay_log": float(departure_delay_log),
        }

        try:
            with st.spinner("Łączenie z API..."):
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=20)

            if response.status_code == 200:
                result = response.json()
                prediction = result.get("prediction")
                model_used = result.get("model_name", "AutoGluon")

                if prediction == 1:
                    st.success("Wynik: pasażer zadowolony (1)")
                else:
                    st.warning("Wynik: pasażer neutralny / niezadowolony (0)")

                st.caption(f"Model: {model_used}")
            elif response.status_code == 422:
                st.error("Błędne dane wejściowe (walidacja Pydantic).")
                st.json(response.json())
            elif response.status_code == 503:
                st.error("API działa, ale model nie jest załadowany. Uruchom kedro run i zrestartuj API.")
            else:
                st.error(f"Błąd API ({response.status_code}): {response.text}")

        except requests.exceptions.Timeout:
            st.error("Przekroczono czas oczekiwania na odpowiedź API. Spróbuj ponownie.")
        except requests.exceptions.ConnectionError:
            st.error(
                f"Nie można połączyć się z API pod {API_URL}. "
                "Upewnij się, że uvicorn api.main:app jest uruchomione."
            )

with tab_data:
    st.header("Podgląd danych")

    df = load_data()

    if df.empty:
        st.warning(f"Brak bazy danych pod ścieżką `{DB_PATH}`.")
    else:
        st.write(f"Liczba rekordów: {len(df)}")
        st.dataframe(df.head(100), use_container_width=True)

        st.subheader("Statystyki opisowe")
        st.dataframe(df.describe(), use_container_width=True)

        st.subheader("Rozkład wybranej kolumny")
        numeric_cols = [c for c in df.select_dtypes("number").columns if c != "ID"]
        if numeric_cols:
            default_idx = numeric_cols.index("Age") if "Age" in numeric_cols else 0
            column = st.selectbox("Kolumna", numeric_cols, index=default_idx)
            if df[column].nunique() > 50:
                st.bar_chart(df[column].value_counts(bins=30, sort=False))
            else:
                st.bar_chart(df[column].value_counts().sort_index())
        else:
            st.info("Brak kolumn numerycznych do wizualizacji.")

with tab_synth:
    st.header("Dane syntetyczne (SDV)")

    real_df = load_data()

    if real_df.empty:
        st.error("Brak danych wejściowych w SQLite.")
    else:
        n_samples = st.number_input("Liczba rekordów do wygenerowania", 100, 10_000, 1000, step=100)

        if st.button("Generuj dane syntetyczne"):
            with st.spinner("Trenowanie syntezatora i generowanie..."):
                sample_size = min(len(real_df), 500)
                df_sample = real_df.sample(n=sample_size, random_state=42)
                synthesizer = fit_synthesizer(df_sample)
                st.session_state["synthetic"] = synthesizer.sample(num_rows=int(n_samples))

        if "synthetic" in st.session_state:
            synthetic = st.session_state["synthetic"]
            st.success(f"Wygenerowano {len(synthetic)} rekordów.")

            col_real, col_synth = st.columns(2)
            with col_real:
                st.subheader("Oryginał (statystyki)")
                st.dataframe(real_df.describe(), use_container_width=True)
            with col_synth:
                st.subheader("Syntetyczne (statystyki)")
                st.dataframe(synthetic.describe(), use_container_width=True)

        st.divider()
        st.subheader("Plik z pipeline Kedro")
        if SYNTHETIC_DATA_PATH.exists():
            kedro_synth_df = pd.read_csv(SYNTHETIC_DATA_PATH)
            st.write(f"`kedro run --pipeline=synthetic` → {SYNTHETIC_DATA_PATH.name}")
            st.dataframe(kedro_synth_df.head(50), use_container_width=True)
        else:
            st.info("Uruchom `kedro run --pipeline=synthetic`, aby wygenerować plik CSV.")
