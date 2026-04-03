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

## Kolejne kroki

