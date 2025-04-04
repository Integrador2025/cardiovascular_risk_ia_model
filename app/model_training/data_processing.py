import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """Preprocesa los datos para el entrenamiento del modelo."""
    # Variables objetivo
    Y_class = df['RIESGO_CARDIOVASCULAR'].astype(int)
    Y_reg = df['PUNTAJE_RIESGO'].astype(float)
    
    # Definir columnas numéricas y categóricas
    numeric_features = ['EDAD', 'IMC', 'SEXO', 'COLESTEROL', 'ESTRATO', 'NIVEL_EDUCATIVO',
                       'ACCESO_ELECTRICO', 'ACUEDUCTO', 'ALCANTARILLADO', 'GAS_NATURAL', 
                       'ANTECEDENTES_FAMILIARES', 'FUMADOR']
    categorical_features = [
        'DEPARTAMENTO', 'MUNICIPIO', 'ESTADO_CIVIL', 'AREA',  
        'INTERNET', 'ETNIA', 'OCUPACION',
    ]
    
    # Eliminar columnas objetivo de las características
    X = df.drop(['RIESGO_CARDIOVASCULAR', 'PUNTAJE_RIESGO'], axis=1)
    
    # Convertir a numérico
    for col in numeric_features:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    for col in categorical_features:
        if X[col].dtype == 'object':
            X[col] = pd.Categorical(X[col]).codes
    
    # Eliminar filas con NaN
    X.dropna(inplace=True)
    # Alinear objetivos con X
    Y_class = Y_class[X.index]
    Y_reg = Y_reg[X.index]
    
    # Escalado de características numéricas
    scaler_numeric = StandardScaler()
    X[numeric_features] = scaler_numeric.fit_transform(X[numeric_features])
    
    # One-hot encoding para variables categóricas
    X = pd.get_dummies(X, columns=categorical_features)
    
    # Extraer nombres de las variables
    feature_names = X.columns.tolist()
    
    # Convertir a arrays de numpy
    X_array = X.values.astype(np.float32)
    Y_class_array = Y_class.values.astype(np.int32)
    Y_reg_array = Y_reg.values.astype(np.float32)
    
    return X_array, (Y_class_array, Y_reg_array), scaler_numeric, feature_names