Projekt realizowany w ramach przedmiotu **Architektury rozwiązań i metodyki wdrożeń SI**.

## Zespół

- s27788 — Lidia Kongiel
- s27841 — Maria Góral
- s26786 — Michał Sobieski
- s24602 — Mateusz Brański
- s25698 — Bartłomiej Kąkol

---

## Cel projektu

Celem projektu jest stworzenie **end-to-end systemu ML w podejściu MLOps**, obejmującego:

- przetwarzanie danych
- pipeline ML
- trening modelu
- ewaluację
- dalszą rozbudowę o API i dashboard

---

## Problem ML

**Klasyfikacja satysfakcji pasażerów linii lotniczych**

- typ: klasyfikacja binarna
- target: `Satisfaction`
- wartości:
    - `1` → satisfied
    - `0` → neutral or dissatisfied

---

## Dane

- źródło: Kaggle
- dataset: Airline Passenger Satisfaction
- https://www.kaggle.com/datasets/mysarahmadbhat/airline-passenger-satisfaction

Dane są przechowywane w **SQLite** i ładowane przez Kedro Data Catalog.

---

## Sprint 2 — Kedro pipeline

Zrealizowane:

- ✔ Projekt Kedro (`kedro new`)
- ✔ Pipeline `data_processing`
- ✔ Node: `preprocess_data`
- ✔ Dane ładowane przez `SQLTableDataset` (SQLite)
- ✔ Zapis wyników do `ParquetDataset`

---

## Preprocessing

W pipeline wykonujemy:

- usunięcie duplikatów
- usunięcie kolumny `ID`
- konwersję targetu `Satisfaction` → `0/1`
- usunięcie 14 kolumn powodujących **data leakage**
- uzupełnienie braków w `Arrival Delay`
- log-transform dla:
    - `Arrival Delay`
    - `Departure Delay`

Finalny dataset:
- shape: `(129880, 9)`

---

## Uruchomienie projektu

### 1. Stwórz środowisko

```bash
conda create -n asi python=3.10
conda activate asi
```

### 2. Zainstaluj zależności

```bash
pip install -r requirements.txt
pip install kedro==0.19.9 kedro-datasets pyarrow
```

### 3. Uruchom pipeline

```bash
kedro run
```

---

## Sprint 4: AutoML (AutoGluon)

### Konfiguracja lokalna

**1. Credentials Kedro** — utwórz `conf/local/credentials.yml` (nie commituj):

```yaml
db_credentials:
  con: sqlite:///data/01_raw/dataset.db
```

**2. Plik `.env`** — skopiuj z `.env.example` i uzupełnij:

```bash
cp .env.example .env
```

Wymagane zmienne: `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY`.

**3. Baza danych** — umieść `dataset.db` w `data/01_raw/` (tabela `airline_data`).

### Uruchomienie

```bash
kedro run --pipeline=full
```

### Eksperymenty AutoGluon

W `conf/base/parameters.yml` zmieniamy tylko sekcję `automl` (`presets`, `time_limit`), potem:

```bash
kedro run --pipeline=automl
```

Przykładowe 3 eksperymenty:

| # | presets | time_limit |
|---|---------|------------|
| 1 | `medium_quality` | 120 |
| 2 | `medium_quality` | 300 |
| 3 | `good_quality` | 300 |

Parametr `automl.clean_path: true` **automatycznie usuwa** katalog `data/06_models/autogluon/` przed treningiem każdy eksperyment startuje od czystego stanu.

### W&B — gdzie szukać wyników

- **Baseline:** run `baseline-random-forest` (metryki `accuracy`, `f1`)
- **AutoGluon:** run `automl-<preset>-<time>s` (metryka `f1`)
- **Leaderboard:** otwórz run AutoGluon → zakładka **Tables** → `leaderboard` (nie Charts)

---
## Wyniki baseline

Model:

- `RandomForestClassifier`

Wyniki:
Accuracy 0.8068
F1 Score 0.7820

Run zapisany w W&B:

```text

baseline-random-forest

```

## Wyniki AutoML

### Porównanie eksperymentów

| Eksperyment | presets | time_limit | Najlepszy model | F1 |
|---|---|---:|---|---:|
| 1 | `medium_quality` | 120 | `ExtraTreesGini` | 0.7729 |
| 2 | `medium_quality` | 300 | `WeightedEnsemble_L2` | 0.7740 |
| 3 | `good_quality` | 300 | `RandomForestGini_BAG_L1` | 0.7766 |

---

## Wnioski z eksperymentów

- zwiększenie `time_limit` z `120s` do `300s` nie przyniosło dużej poprawy jakości modelu
- preset `good_quality` osiągnął najlepszy wynik spośród eksperymentów AutoML
- najlepszy wynik AutoML: F1 = 0.7766
- najlepszy model AutoML to RandomForestGini_BAG_L1
- Baseline `RandomForestClassifier` nadal osiągnął nieco lepszy wynik (`F1 ≈ 0.782`)
- eksperymenty pokazały wpływ parametrów `presets` oraz `time_limit` na jakość modeli

## Screenshots

Screenshoty z eksperymentów i dashboardów W&B znajdują się w:

```text

docs/screenshots_sprint04/

```

Zawierają:

- terminale z uruchomienia pipeline’ów

- dashboardy W&B

- leaderboardy modeli AutoGluon

- porównanie eksperymentów AutoML (`wandb_experiments_overview.png`)

## Sprint 5: FastAPI

### Cel

Przygotowanie REST API umożliwiającego wykonywanie predykcji przy użyciu wytrenowanego modelu AutoGluon.  
API działa niezależnie od pipeline'ów Kedro i wykorzystuje model zapisany na dysku.

### Wymagania

Przed uruchomieniem API model musi zostać wcześniej wytrenowany i zapisany w katalogu:

```text
data/06_models/autogluon/
```

Jeżeli model nie istnieje, należy uruchomić:

```bash
kedro run --pipeline=full
```

lub:

```bash
kedro run --pipeline=automl
```

### Uruchomienie

```bash
conda activate asi
uvicorn api.main:app --reload
```

Dokumentacja Swagger:

```text
http://127.0.0.1:8000/docs
```

### Zrealizowane

- utworzenie aplikacji FastAPI
- ladowanie modelu AutoGluon przy starcie aplikacji
- endpoint GET /health
- endpoint GET /model-info
- endpoint POST /predict
- walidacja danych wejściowych przy użyciu Pydantic
- obsługa błędów HTTP (422, 500, 503)
- automatyczna dokumentacja Swagger

### Endpointy

| Metoda | Endpoint | Opis |
|---------|---------|---------|
| GET | /health | Status API oraz informacja o załadowanym modelu |
| GET | /model-info | Informacje o aktualnie załadowanym modelu |
| POST | /predict | Wykonanie predykcji |

### Dane wejściowe

Model wykorzystuje cechy:

- Gender
- Age
- Customer Type
- Type of Travel
- Class
- Flight Distance
- arrival_delay_log
- departure_delay_log

pola arrival_delay_log oraz departure_delay_log zawierają wartości po transformacji log1p()  
wykonanej podczas preprocessingu danych.

Przykłady:

- brak opóźnienia -> 0.0
- 1 minuta opóźnienia -> ~0.69

### Przykladowe wywolanie

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Gender": "Female",
    "Age": 35,
    "Customer Type": "Returning",
    "Type of Travel": "Business",
    "Class": "Business",
    "Flight Distance": 1200,
    "arrival_delay_log": 0.69,
    "departure_delay_log": 1.10
  }'
```


## Sprint 6: Integracja Systemu (Streamlit, FastAPI & SDV)

### Cel

Zintegrowanie wszystkich elementów systemu w architekturze end-to-end.
Dodanie potoku generowania danych syntetycznych przy użyciu SDV wraz z logowaniem metryk do Weights & Biases oraz stworzenie interaktywnego dashboardu w Streamlit komunikującego się z API FastAPI.

### Wymagania

Przed uruchomieniem systemu należy:

* posiadać wytrenowany model (pipeline `full` lub `automl`)
* skonfigurować zmienne środowiskowe i sekrety

Wymagany plik:

```text
.streamlit/secrets.toml
```

Zawartość:

```toml
API_URL = "http://127.0.0.1:8000"
```

### Uruchomienie

#### Pipeline danych syntetycznych

```bash
kedro run --pipeline=synthetic
```

Wynik zapisywany jest w:

```text
data/03_primary/synthetic_data.csv
```

#### Uruchomienie pełnego systemu

Backend (FastAPI):

```bash
uvicorn api.main:app --reload
```

Frontend (Streamlit):

```bash
streamlit run app/streamlit_app.py
```

### Zrealizowane

* integracja FastAPI ze Streamlit (komunikacja HTTP)
* stworzenie pipeline `synthetic` w Kedro
* generowanie danych syntetycznych przy użyciu SDV
* logowanie metryk `diagnostic_score` oraz `quality_score` do Weights & Biases
* cache’owanie danych w Streamlit (`@st.cache_data`, `@st.cache_resource`)
* obsługa błędów sieciowych i timeoutów w komunikacji z API
* interaktywny dashboard użytkownika

### Moduły aplikacji (Streamlit)

#### Predykcja Satysfakcji

* formularz wejściowy (slidery, selectboxy)
* wysyłanie zapytań POST do API (`/predict`)
* obsługa błędów i timeoutów
* prezentacja wyniku predykcji

#### Przegląd Danych Realnych

* ładowanie danych w formacie Parquet
* cache’owanie danych
* podgląd próbek danych
* statystyki opisowe (`.describe()`)

#### Generator Danych Syntetycznych

* wykorzystanie `GaussianCopulaSynthesizer`
* trenowanie lekkiego modelu w locie
* generowanie dowolnej liczby rekordów
* podgląd wcześniej wygenerowanego pliku z pipeline Kedro

### Przepływ systemu

1. Użytkownik wprowadza dane w aplikacji Streamlit
2. Aplikacja wysyła zapytanie do API FastAPI
3. API wykorzystuje model AutoGluon do predykcji
4. Wynik zwracany jest do frontend i prezentowany użytkownikowi
5. Niezależnie pipeline `synthetic` generuje i ocenia dane syntetyczne
