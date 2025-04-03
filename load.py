import pandas as pd

def load_dataset():
    # Cargar dataset de pacientes
    df_pacientes = pd.read_csv("datasets/pacient_dataset2.csv")

    # Cargar dataset de códigos municipios y departamentos
    df_codigos = pd.read_csv("datasets/codigo_departamentos_dataset.csv")

    # Cargar dataset de información geográfica
    df_departamentos = pd.read_csv("datasets/departamentos_dataset.csv")

    return df_pacientes, df_codigos, df_departamentos