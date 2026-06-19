import pandas as pd
import wandb
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.evaluation.single_table import run_diagnostic, evaluate_quality

def generate_synthetic_data(real_data: pd.DataFrame, parameters: dict) -> pd.DataFrame:
    """
    Wykrywa metadane przy użyciu Metadata.detect_from_dataframe(),
    trenuje model GaussianCopulaSynthesizer i generuje dane syntetyczne.
    
    Args:
        real_data: Oryginalny zbiór danych (Pandas DataFrame).
        parameters: Słownik z parametrami.
        
    Returns:
        pd.DataFrame: Wygenerowane dane syntetyczne.
    """
    # Aktualny sposób wykrywania metadanych (zgodnie z instrukcją)
    metadata = Metadata.detect_from_dataframe(data=real_data)
    
    # Inicjalizacja i uczenie syntezatora
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(real_data)
    
    # Próbkowanie/Generowanie danych
    num_rows = parameters.get("synthetic_rows", 100)
    synthetic_data = synthesizer.sample(num_rows=num_rows)
    
    return synthetic_data

def evaluate_synthetic_data(real_data: pd.DataFrame, synthetic_data: pd.DataFrame, parameters: dict) -> dict:
    """
    Dokonuje oceny jakości przy użyciu run_diagnostic i evaluate_quality,
    a następnie loguje wyniki do platformy Weights & Biases.
    
    Args:
        real_data: Oryginalne dane.
        synthetic_data: Wygenerowane dane sztuczne.
        parameters: Słownik parametrów.
        
    Returns:
        dict: Słownik z metrykami.
    """
    metadata = Metadata.detect_from_dataframe(data=real_data)
    
    # Wyczyszczenie ewentualnych braków dla poprawnego działania metryk
    real_clean = real_data.dropna()
    synth_clean = synthetic_data.dropna()
    
    # Ewaluacja SDV
    diagnostic = run_diagnostic(real_data=real_clean, synthetic_data=synth_clean, metadata=metadata)
    quality = evaluate_quality(real_data=real_clean, synthetic_data=synth_clean, metadata=metadata)
    
    diagnostic_score = diagnostic.get_score()
    quality_score = quality.get_score()
    
    # Logowanie do Weights & Biases
    wandb.init(
        project=parameters.get("wandb_project", "asi-pjatk"),
        job_type="sdv-evaluation"
    )
    
    metrics = {
        "diagnostic_score": float(diagnostic_score),
        "quality_score": float(quality_score)
    }
    
    wandb.log(metrics)
    wandb.finish()
    
    return metrics