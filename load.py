import pandas as pd

def load_dataset():
    # Cargar dataset de pacientes
    df_pacientes = pd.read_csv("datasets/pacient_dataset.csv")

    # Cargar dataset de códigos municipios y departamentos
    df_municipios = pd.read_csv("datasets/municipios_colombia.csv")

    return df_pacientes, df_municipios, True