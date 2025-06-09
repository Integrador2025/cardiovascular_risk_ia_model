import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error
from app.model_training.model_architecture import create_regression_model
from app.model_training.utils import (
    save_model,
    plot_metrics,
    save_feature_importances,
    plot_predictions_vs_true,
    plot_residuals,
    plot_mse,
    plot_prediction_distribution,
    plot_residuals_vs_predictions,
    plot_error_by_range
)
from app.model_training.data_processing import preprocess_data
import joblib
from app.model_config import (
    INDIVIDUAL_MODEL_PATH,
    INDIVIDUAL_SCALER_PATH,
    INDIVIDUAL_FEATURE_NAMES_PATH
)

def train_individual_model(df_pacientes):
    """
    Entrena el modelo de riesgo individual, guarda el modelo, el scaler y los nombres de las características.

    Args:
        df_pacientes (pd.DataFrame): DataFrame con los datos de pacientes.

    Returns:
        tuple: (final_model, fitted_scaler, final_feature_names, history) del entrenamiento.
    """

    # Preprocesar datos
    X, Y_reg, fitted_scaler, final_feature_names = preprocess_data(df_pacientes)

    # Crear bins para estratificación
    bins = np.digitize(Y_reg, bins=[0.0, 0.3, 0.5, 0.7, 0.9, 1.1])

    # Dividir en entrenamiento y validación
    X_train_final, X_val_final, Y_train_final, Y_val_final = train_test_split(
        X, Y_reg, test_size=0.2, random_state=42, stratify=bins
    )

    print("🧠 Construyendo modelo final...")
    final_model = create_regression_model(X_train_final.shape[1])

    # Asegúrate de compilar el modelo con MSE como métrica
    final_model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mean_squared_error']
    )

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
        epochs=400,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )

    # Verificar o crear carpeta "model" para modelos, scaler y nombres de características
    model_dir = "model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Carpeta creada: {model_dir}")
    else:
        print(f"Carpeta ya existe: {model_dir}")

    # Verificar o crear carpeta "metrics" para gráficos
    metrics_dir = "metrics"
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)
        print(f"Carpeta creada: {metrics_dir}")
    else:
        print(f"Carpeta ya existe: {metrics_dir}")

    # Guardar modelo, scaler y nombres de características en "model"
    save_model(final_model, os.path.basename(INDIVIDUAL_MODEL_PATH).replace(".keras", ""))
    joblib.dump(fitted_scaler, INDIVIDUAL_SCALER_PATH)
    joblib.dump(final_feature_names, INDIVIDUAL_FEATURE_NAMES_PATH)

    # Gráfico de pérdida
    plot_path = os.path.join(metrics_dir, "loss_final_regresion.png")
    plot_metrics(history, "loss", plot_path, branch='')
    print(f"Verificando si se generó: {os.path.exists(plot_path)}")

    # Gráfico de MSE
    mse_plot_path = os.path.join(metrics_dir, "mse_final_regresion.png")
    plot_mse(history, mse_plot_path)
    print(f"Verificando si se generó: {os.path.exists(mse_plot_path)}")

    # Importancia de características
    feat_path = os.path.join(metrics_dir, "feature_importances_final.csv")
    save_feature_importances(final_model, final_feature_names, feat_path)
    print(f"Verificando si se generó: {os.path.exists(feat_path)}")

    # Gráficas adicionales
    print("📊 Generando gráficas adicionales...")
    Y_pred_val = final_model.predict(X_val_final).flatten()

    # Pred vs real
    pred_plot_path = os.path.join(metrics_dir, "pred_vs_real.png")
    plot_predictions_vs_true(Y_val_final, Y_pred_val, pred_plot_path)
    print(f"Verificando si se generó: {os.path.exists(pred_plot_path)}")

    # Histograma de residuos
    residual_plot_path = os.path.join(metrics_dir, "residuals_hist.png")
    plot_residuals(Y_val_final, Y_pred_val, residual_plot_path)
    print(f"Verificando si se generó: {os.path.exists(residual_plot_path)}")

    # Distribución de predicciones
    dist_plot_path = os.path.join(metrics_dir, "prediction_distribution.png")
    plot_prediction_distribution(Y_val_final, Y_pred_val, dist_plot_path)
    print(f"Verificando si se generó: {os.path.exists(dist_plot_path)}")

    # Residuos vs predicciones
    resid_vs_pred_path = os.path.join(metrics_dir, "residuals_vs_predictions.png")
    plot_residuals_vs_predictions(Y_val_final, Y_pred_val, resid_vs_pred_path)
    print(f"Verificando si se generó: {os.path.exists(resid_vs_pred_path)}")

    # Error absoluto por rango
    error_range_path = os.path.join(metrics_dir, "error_by_range.png")
    plot_error_by_range(Y_val_final, Y_pred_val, error_range_path)
    print(f"Verificando si se generó: {os.path.exists(error_range_path)}")

    return final_model, fitted_scaler, final_feature_names, history