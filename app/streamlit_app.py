import os
import requests
import pandas as pd
import streamlit as st
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer
from pathlib import Path

# ------------------------------------------------------------------
# CONFIGURATION & CONSTANTS
# ------------------------------------------------------------------
# Dynamiczne wyznaczanie ścieżki głównej projektu niezależnie od miejsca uruchomienia
APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent

# Bezpieczne pobranie adresu API z konfiguracji Streamlita / zmiennych środowiskowych
API_URL = os.getenv("API_URL", "http://localhost:8000")

# Ścieżki do plików z danymi jako obiekty klasy Path
REAL_DATA_PATH = BASE_DIR / "data" / "02_intermediate" / "processed_data.parquet"
SYNTHETIC_DATA_PATH = BASE_DIR / "data" / "03_primary" / "synthetic_data.csv"

st.set_page_config(
    page_title="Airline Satisfaction & SDV System",
    layout="wide"
)

st.title("Airline Satisfaction - Panel Kontrolny")


# ------------------------------------------------------------------
# CACHED FUNCTIONS
# ------------------------------------------------------------------

@st.cache_data
def load_real_data(path: Path) -> pd.DataFrame:
    """Wczytuje i cache'uje oryginalne dane (Zakładka Dane czyta z pliku Parquet)."""
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=["Gender", "Age", "Customer Type", "Type of Travel", "Class", "Flight Distance"])


@st.cache_resource
def get_cached_synthesizer(real_data_sample: pd.DataFrame) -> GaussianCopulaSynthesizer:
    """Inicjalizuje, trenuje i cache'uje syntezator SDV w Streamlit."""
    metadata = Metadata.detect_from_dataframe(data=real_data_sample)
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(real_data_sample)
    return synthesizer


# Definicja zakładek interfejsu użytkownika
tab_predict, tab_data, tab_synthetic = st.tabs([
    "Predykcja Satysfakcji",
    "Przegląd Danych Realnych",
    "Generator Danych Syntetycznych"
])

# ------------------------------------------------------------------
# ZAKŁADKA 1: PREDYKCJA SATYSFAKCJI (KOMUNIKACJA Z FASTAPI)
# ------------------------------------------------------------------
with tab_predict:
    st.header("Formularz predykcji satysfakcji pasażera")
    st.markdown("Wprowadź parametry lotu, aby sprawdzić, czy pasażer będzie usatysfakcjonowany.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender (Płeć)", options=["Female", "Male"])
            age = st.number_input("Age (Wiek)", min_value=1, max_value=120, value=35)
            customer_type = st.selectbox("Customer Type (Typ klienta)", options=["Returning", "First-time"])

        with col2:
            type_of_travel = st.selectbox("Type of Travel (Cel podróży)", options=["Business", "Personal"])
            ticket_class = st.selectbox("Class (Klasa biletu)", options=["Business", "Economy", "Economy Plus"])
            flight_distance = st.number_input("Flight Distance (Dystans w milach)", min_value=1, max_value=10000,
                                              value=1200)

        with col3:
            arrival_delay_log = st.slider("Arrival Delay - log1p(minutes)", min_value=0.0, max_value=10.0, value=0.69,
                                          step=0.01)
            departure_delay_log = st.slider("Departure Delay - log1p(minutes)", min_value=0.0, max_value=10.0,
                                            value=1.10, step=0.01)

        submit_button = st.form_submit_button("Wyślij zapytanie do API")

    if submit_button:
        payload = {
            "Gender": gender,
            "Age": int(age),
            "Customer Type": customer_type,
            "Type of Travel": type_of_travel,
            "Class": ticket_class,
            "Flight Distance": int(flight_distance),
            "arrival_delay_log": float(arrival_delay_log),
            "departure_delay_log": float(departure_delay_log)
        }

        try:
            with st.spinner("Łączenie z backendem FastAPI..."):
                # Timeout wydłużony do 20 sekund, aby uniknąć błędów przy starcie modelu
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=20)

            if response.status_code == 200:
                result = response.json()
                prediction = result.get("prediction")
                model_used = result.get("model_name", "AutoGluon")

                if prediction == 1:
                    st.success(f"**Wynik predykcji: Pasażer SATYSFAKCJONOWANY (1)**")
                else:
                    st.warning(f"**Wynik predykcji: Pasażer NEUTRALNY / NIEZADOWOLONY (0)**")

                st.caption(f"Model użyty do predykcji backendowej: `{model_used}`")
            else:
                st.error(f"API zwróciło kod błędu: {response.status_code}. Szczegóły: {response.text}")

        except requests.exceptions.Timeout:
            st.error("Serwer potrzebuje więcej czasu na rozgrzanie modelu. Spróbuj kliknąć przycisk ponownie!")
        except requests.exceptions.ConnectionError:
            st.error(
                f"Nie można połączyć się z API pod adresem: {API_URL}. Upewnij się, że serwer FastAPI (Uvicorn) został uruchomiony!")
        except Exception as e:
            st.error(f"Wystąpił nieoczekiwany błąd komunikacji: {e}")

# ------------------------------------------------------------------
# ZAKŁADKA 2: PRZEGLĄD DANYCH REALNYCH
# ------------------------------------------------------------------
with tab_data:
    st.header("Przegląd oryginalnego zbioru danych")

    real_df = load_real_data(REAL_DATA_PATH)

    if not real_df.empty:
        st.write(f"Łączna liczba wierszy w pliku bazowym: `{len(real_df)}`")
        st.dataframe(real_df.head(100), use_container_width=True)

        st.subheader("Podstawowe statystyki opisowe")
        st.dataframe(real_df.describe(), use_container_width=True)
    else:
        st.warning(
            f"Nie znaleziono pliku danych pod ścieżką: `{REAL_DATA_PATH}`. Upewnij się, że potoki Kedro zostały uruchomione.")

# ------------------------------------------------------------------
# ZAKŁADKA 3: GENERATOR DANYCH SYNTETYCZNYCH (SDV W LOCIE + KEDRO)
# ------------------------------------------------------------------
with tab_synthetic:
    st.header("Generowanie danych sztucznych (SDV)")
    st.markdown("Ta sekcja korzysta z cache'owanego syntezatora `GaussianCopulaSynthesizer` trenowanego w locie.")

    real_df_for_sdv = load_real_data(REAL_DATA_PATH)

    if not real_df_for_sdv.empty:
        # Ograniczamy próbkę do treningu w locie na froncie, by Streamlit działał wydajnie i szybko
        sample_size = min(len(real_df_for_sdv), 500)
        df_sample = real_df_for_sdv.sample(n=sample_size, random_state=42)

        with st.spinner("Przygotowywanie modelu syntezatora w pamięci podręcznej..."):
            # Wywołanie funkcji z dekoratorem @st.cache_resource
            synthesizer = get_cached_synthesizer(df_sample)

        st.success("Syntezator SDV pomyślnie załadowany do pamięci frontu!")

        st.subheader("Wygeneruj nowe rekordy")
        num_rows_to_generate = st.slider("Wybierz liczbę wierszy do wygenerowania:", min_value=10, max_value=200,
                                         value=50)

        if st.button("Wygeneruj dane sztuczne"):
            with st.spinner("Generowanie rekordów za pomocą SDV..."):
                # Próbkowanie z cache'owanego syntezatora
                gen_data = synthesizer.sample(num_rows=num_rows_to_generate)
            st.dataframe(gen_data, use_container_width=True)

        # Opcjonalny wgląd w trwały plik wyjściowy pipeline z Kedro
        st.divider()
        st.subheader("Podgląd trwałego pliku wygenerowanego wcześniej przez Kedro")
        if SYNTHETIC_DATA_PATH.exists():
            kedro_synth_df = pd.read_csv(SYNTHETIC_DATA_PATH)
            st.write(f"Dane z potoku `kedro run --pipeline=synthetic` (`{SYNTHETIC_DATA_PATH}`):")
            st.dataframe(kedro_synth_df.head(50), use_container_width=True)
        else:
            st.info(
                f"Brak pliku `{SYNTHETIC_DATA_PATH.name}`. Uruchom potok Kedro (`kedro run --pipeline=synthetic`), aby go stworzyć.")
    else:
        st.error("Brak danych wejściowych do uruchomienia syntezatora.")