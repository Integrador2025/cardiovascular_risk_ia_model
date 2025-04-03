import os
import numpy as np
import pandas as pd
import load
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score,
    mean_squared_error
)
import matplotlib.pyplot as plt

# Preprocesamiento de datos y extracción de variables y objetivos
def preprocess_data(df):
    # Variables objetivo:
    # - Para clasificación: "RIESGO_CARDIOVASCULAR" (enteros 0 a 4)
    # - Para regresión: "PUNTAJE_RIESGO" (puntaje continuo)
    Y_class = df['RIESGO_CARDIOVASCULAR'].astype(int)
    Y_reg = df['PUNTAJE_RIESGO'].astype(float)
    
    # Definir columnas numéricas y categóricas (se eliminan las columnas de objetivo)
    numeric_features = ['EDAD', 'IMC', 'SEXO', 'COLESTEROL', 'ESTRATO', 'NIVEL_EDUCATIVO',
                        'ACCESO_ELECTRICO', 'ACUEDUCTO', 'ALCANTARILLADO', 'GAS_NATURAL', 'ANTECEDENTES_FAMILIARES', 'FUMADOR']
    categorical_features = [
        'DEPARTAMENTO', 'MUNICIPIO', 'ESTADO_CIVIL', 'AREA',  
        'INTERNET', 'ETNIA', 'OCUPACION',
    ]
    
    # Eliminar columnas objetivo de las características
    X = df.drop(['RIESGO_CARDIOVASCULAR', 'PUNTAJE_RIESGO'], axis=1)
    
    # Convertir a numérico las características numéricas y mapear las categóricas
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
    
    # Extraer nombres de las variables (columnas)
    feature_names = X.columns.tolist()
    
    # Convertir a arrays de numpy
    X_array = X.values.astype(np.float32)
    Y_class_array = Y_class.values.astype(np.int32)
    Y_reg_array = Y_reg.values.astype(np.float32)
    
    # Devolver X y una tupla con ambos objetivos
    return X_array, (Y_class_array, Y_reg_array), scaler_numeric, feature_names

# Construcción del modelo multitarea con mayor capacidad y Dropout
def create_multitask_model(input_shape):
    inputs = tf.keras.layers.Input(shape=(input_shape,))
    
    # Capas compartidas con más neuronas
    x = tf.keras.layers.Dense(
        2048, activation='relu', 
        kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4)
    )(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    
    x = tf.keras.layers.Dense(
        128, activation='relu', 
        kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    x = tf.keras.layers.Dense(
        512, activation='relu', 
        kernel_regularizer=tf.keras.regularizers.L1L2(l1=1e-5, l2=1e-4)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Rama para clasificación (5 clases de riesgo)
    clasificacion = tf.keras.layers.Dense(5, activation='softmax', name='clasificacion')(x)
    
    # Rama para regresión (puntaje continuo)
    regresion = tf.keras.layers.Dense(1, activation='linear', name='regresion')(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=[clasificacion, regresion])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss={
            'clasificacion': 'sparse_categorical_crossentropy',
            'regresion': 'mean_squared_error'
        },
        metrics={
            'clasificacion': ['accuracy'],
            'regresion': ['mse']
        }
    )
    
    return model

# Función para guardar el modelo
def save_model(model, model_name="modelo_entrenado"):
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}.h5")
    model.save(model_path)
    print(f"Modelo guardado en: {model_path}")

# Función para guardar gráficos de las métricas
def plot_metrics(history, metric, save_path, branch='clasificacion'):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history[f"{branch}_{metric}"], label=f"train {metric}")
    plt.plot(history.history[f"val_{branch}_{metric}"], label=f"val {metric}")
    plt.xlabel("Epochs")
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.title(f"Evolución de {metric} ({branch})")
    plt.grid()
    plt.savefig(save_path)
    plt.close()
    print(f"Gráfico guardado en: {save_path}")

# Función para guardar “importancia” de las variables basada en los pesos de la primera capa densa con pesos
def save_feature_importances(model, feature_names, save_path):
    # Buscar la primera capa con pesos (omitiendo la capa de entrada)
    first_dense = None
    for layer in model.layers:
        if hasattr(layer, 'kernel'):
            first_dense = layer
            break
    if first_dense is None:
        print("No se encontró una capa densa con pesos.")
        return
    # Extraer los pesos de la capa encontrada
    kernel = first_dense.get_weights()[0]  # shape: (input_shape, neuronas)
    importance = np.sum(np.abs(kernel), axis=1)
    # Combinar nombre de variable con su importancia
    feat_imp = list(zip(feature_names, importance))
    # Ordenar de mayor a menor importancia
    feat_imp.sort(key=lambda x: x[1], reverse=True)
    # Guardar en archivo CSV
    with open(save_path, 'w') as f:
        f.write("Feature,Importance\n")
        for feat, imp in feat_imp:
            f.write(f"{feat},{imp}\n")
    print(f"Importancia de variables guardada en: {save_path}")

# Entrenamiento con validación cruzada estratificada basado en la variable de clasificación
def train_with_stratified_kfold(X, Y, feature_names, n_splits=20):
    Y_class, Y_reg = Y  # Separar objetivos
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for fold, (train_index, val_index) in enumerate(skf.split(X, Y_class), 1):
        print(f"\nFold {fold}")
        X_train, X_val = X[train_index], X[val_index]
        Y_train_class, Y_train_reg = Y_class[train_index], Y_reg[train_index]
        Y_val_class, Y_val_reg = Y_class[val_index], Y_reg[val_index]
        
        # Callbacks para early stopping y reducción de learning rate
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_clasificacion_loss', 
            patience=15, 
            restore_best_weights=True,
            mode='min'
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_clasificacion_loss', 
            factor=0.2, 
            patience=5, 
            min_lr=1e-5,
            mode='min'
        )
        
        # Crear y entrenar el modelo multitarea
        model = create_multitask_model(X_train.shape[1])
        history = model.fit(
            X_train, 
            {'clasificacion': Y_train_class, 'regresion': Y_train_reg},
            validation_data=(X_val, {'clasificacion': Y_val_class, 'regresion': Y_val_reg}),
            epochs=200,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        # Guardar el modelo de este fold
        save_model(model, f"modelo_fold_{fold}")
        
        # Guardar gráficos de la evolución de la pérdida y la precisión para la rama de clasificación
        os.makedirs("model", exist_ok=True)
        plot_metrics(history, "loss", f"model/loss_fold_{fold}.png", branch='clasificacion')
        plot_metrics(history, "accuracy", f"model/accuracy_fold_{fold}.png", branch='clasificacion')
        
        # Predicciones y cálculo de métricas
        preds = model.predict(X_val)
        # Rama de clasificación: predicción de probabilidades y etiqueta mediante argmax
        Y_pred_proba = preds[0]
        Y_pred_class = np.argmax(Y_pred_proba, axis=1)
        # Rama de regresión: predicción del puntaje continuo
        Y_pred_reg = preds[1].ravel()
        
        print("\nInforme de Clasificación:")
        print(classification_report(Y_val_class, Y_pred_class))
        try:
            auc_roc = roc_auc_score(Y_val_class, Y_pred_proba, multi_class='ovr')
            print(f"AUC-ROC para Fold {fold}: {auc_roc:.4f}")
        except Exception as e:
            print(f"No se pudo calcular AUC-ROC: {e}")
        
        precision = precision_score(Y_val_class, Y_pred_class, average='weighted', zero_division=0)
        recall = recall_score(Y_val_class, Y_pred_class, average='weighted', zero_division=0)
        f1 = f1_score(Y_val_class, Y_pred_class, average='weighted', zero_division=0)
        
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
        
        # Métricas de regresión: calcular MSE
        mse_reg = mean_squared_error(Y_val_reg, Y_pred_reg)
        print(f"MSE (Regresión) para Fold {fold}: {mse_reg:.4f}")
        
        # Guardar las “importancias” de las variables del primer fold
        if fold == 1:
            save_feature_importances(model, feature_names, f"model/feature_importances_fold_{fold}.csv")
    
    return model

# Función principal
def main():
    # Cargar datos
    df_pacientes, _, _ = load.load_dataset()
    
    # Preprocesar datos
    X, Y, scaler, feature_names = preprocess_data(df_pacientes)
    
    # Entrenar el modelo multitarea con validación cruzada
    train_with_stratified_kfold(X, Y, feature_names)

if __name__ == '__main__':
    main()
