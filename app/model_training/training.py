# app/model_training/train.py

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error
from app.model_training.model_architecture import create_regression_model
from app.model_training.utils import save_model, plot_metrics, save_feature_importances
from app.model_training.data_processing import preprocess_data # Importar la función actualizada
import joblib # Importar joblib para guardar el scaler y feature_names
from app.model_config import INDIVIDUAL_MODEL_PATH, INDIVIDUAL_SCALER_PATH, INDIVIDUAL_FEATURE_NAMES_PATH

def train_individual_model(df_pacientes):
    """
    Entrena el modelo de riesgo individual, guarda el modelo, el scaler y los nombres de las características.

    Args:
        df_pacientes (pd.DataFrame): DataFrame con los datos de pacientes.

    Returns:
        tuple: (final_model, fitted_scaler, final_feature_names, history) del entrenamiento.
    """
    
    # Preprocesar datos para el modelo individual
    # Ahora preprocess_data devuelve el scaler y los feature_names
    X, Y_reg, fitted_scaler, final_feature_names = preprocess_data(df_pacientes)
    
    # Crear etiquetas simuladas para estratificación por puntaje (usamos bins)
    bins = np.digitize(Y_reg, bins=[0.0, 0.3, 0.5, 0.7, 0.9, 1.1])
    
    # Dividir datos para el entrenamiento final
    X_train_final, X_val_final, Y_train_final, Y_val_final = train_test_split(
        X, Y_reg, test_size=0.2, random_state=42, stratify=bins
    )

    print("🧠 Construyendo modelo final...")
    final_model = create_regression_model(X_train_final.shape[1])

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        mode='min'
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6
    )

    print("🚀 Entrenando modelo final con todos los datos...")
    history = final_model.fit(
        X_train_final, Y_train_final,
        validation_data=(X_val_final, Y_val_final),
        epochs=400, # Reducir epochs para el ejemplo, ajustar según necesidad
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # Verificar o crear carpeta "model"
    model_dir = "model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Carpeta creada: {model_dir}")
    else:
        print(f"Carpeta ya existe: {model_dir}")

    # Guardar el modelo, scaler y feature_names
    save_model(final_model, os.path.basename(INDIVIDUAL_MODEL_PATH).replace(".keras", "")) # save_model espera solo el nombre base
    joblib.dump(fitted_scaler, INDIVIDUAL_SCALER_PATH)
    joblib.dump(final_feature_names, INDIVIDUAL_FEATURE_NAMES_PATH)
    
    # Generar y guardar gráficos y la importancia de las características
    plot_path = os.path.join(model_dir, "loss_final_regresion.png")
    plot_metrics(history, "loss", plot_path, branch='')
    print(f"Verificando si se generó: {os.path.exists(plot_path)}")

    feat_path = os.path.join(model_dir, "feature_importances_final.csv")
    save_feature_importances(final_model, final_feature_names, feat_path)
    print(f"Verificando si se generó: {os.path.exists(feat_path)}")

    return final_model, fitted_scaler, final_feature_names, history