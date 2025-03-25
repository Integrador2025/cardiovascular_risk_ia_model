import load
import pandas as pd

# Cargar datasets
df_pacientes, df_codigo_departamento, df_departamento = load.load_dataset()


# Mostrar las primeras filas de cada dataset
print(df_pacientes.head())
print(df_codigo_departamento.head())
print(df_departamento.head())

