import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import pandas as pd

def save_model(model, model_name="modelo_entrenado"):
    """Guarda el modelo en disco usando el formato .keras recomendado."""
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}.keras")
    model.save(model_path)
    print(f"Modelo guardado en formato Keras moderno en: {model_path}")

def plot_metrics(history, metric, save_path, branch='clasificacion'):
    """Genera y guarda gráficos de métricas de entrenamiento."""
    metrics_dir = os.path.dirname(save_path)
    os.makedirs(metrics_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    key = f"{branch}_{metric}" if branch else metric
    plt.plot(history.history[key], label=f"train {metric}")
    plt.plot(history.history[f"val_{metric}"], label=f"val {metric}")
    plt.xlabel("Epochs")
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.title(f"Evolución de {metric} ({branch})")
    plt.grid()
    plt.savefig(save_path)
    plt.close()
    print(f"Gráfico guardado en: {save_path}")

def save_feature_importances(model, feature_names, save_path):
    """Calcula y guarda la importancia de las características."""
    metrics_dir = os.path.dirname(save_path)
    os.makedirs(metrics_dir, exist_ok=True)
    first_dense = None
    for layer in model.layers:
        if hasattr(layer, 'kernel'):
            first_dense = layer
            break
    if first_dense is None:
        print("No se encontró una capa densa con pesos.")
        return
    
    kernel = first_dense.get_weights()[0]
    importance = np.sum(np.abs(kernel), axis=1)
    feat_imp = list(zip(feature_names, importance))
    feat_imp.sort(key=lambda x: x[1], reverse=True)
    
    with open(save_path, 'w') as f:
        f.write("Feature,Importance\n")
        for feat, imp in feat_imp:
            f.write(f"{feat},{imp}\n")
    print(f"Importancia de variables guardada en: {save_path}")

def plot_predictions_vs_true(y_true, y_pred, save_path):
    """Genera y guarda gráfico de predicciones vs valores reales."""
    metrics_dir = os.path.dirname(save_path)
    os.makedirs(metrics_dir, exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.4)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel("Valor Real")
    plt.ylabel("Predicción")
    plt.title("Predicciones vs Valores Reales")
    plt.grid()
    plt.savefig(save_path)
    plt.close()
    print(f"Gráfico guardado en: {save_path}")

def plot_residuals(y_true, y_pred, save_path):
    """Genera y guarda histograma de residuos."""
    metrics_dir = os.path.dirname(save_path)
    os.makedirs(metrics_dir, exist_ok=True)
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=30, edgecolor='black')
    plt.title("Distribución de Errores (Residuos)")
    plt.xlabel("Error")
    plt.ylabel("Frecuencia")
    plt.grid()
    plt.savefig(save_path)
    plt.close()
    print(f"Gráfico guardado en: {save_path}")

def plot_mse(history, save_path):
    """Genera y guarda gráfico de MSE para entrenamiento y validación."""
    metrics_dir = os.path.dirname(save_path)
    os.makedirs(metrics_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['mean_squared_error'], label='train MSE')
    plt.plot(history.history['val_mean_squared_error'], label='val MSE')
    plt.xlabel("Epochs")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.title("Evolución del Error Cuadrático Medio")
    plt.grid()
    plt.savefig(save_path)
    plt.close()
    print(f"Gráfico de MSE guardado en: {save_path}")

def plot_prediction_distribution(y_true, y_pred, save_path):
    """Genera y guarda histogramas superpuestos de valores reales y predichos."""
    metrics_dir = os.path.dirname(save_path)
    os.makedirs(metrics_dir, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.hist(y_true, bins=30, alpha=0.5, label='Valores Reales', edgecolor='black')
    plt.hist(y_pred, bins=30, alpha=0.5, label='Predicciones', edgecolor='black')
    plt.xlabel("Valor")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de Valores Reales vs Predicciones")
    plt.legend()
    plt.grid()
    plt.savefig(save_path)
    plt.close()
    print(f"Gráfico de distribución guardado en: {save_path}")

def plot_residuals_vs_predictions(y_true, y_pred, save_path):
    """Genera y guarda gráfico de residuos vs predicciones."""
    metrics_dir = os.path.dirname(save_path)
    os.makedirs(metrics_dir, exist_ok=True)
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.4)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel("Predicciones")
    plt.ylabel("Residuos (Real - Predicho)")
    plt.title("Residuos vs Predicciones")
    plt.grid()
    plt.savefig(save_path)
    plt.close()
    print(f"Gráfico de residuos vs predicciones guardado en: {save_path}")

def plot_error_by_range(y_true, y_pred, save_path, bins=5):
    """Genera y guarda boxplot del error absoluto por rangos de valores reales."""
    metrics_dir = os.path.dirname(save_path)
    os.makedirs(metrics_dir, exist_ok=True)
    errors = np.abs(y_true - y_pred)
    df = pd.DataFrame({'y_true': y_true, 'error': errors})
    df['range'] = pd.cut(df['y_true'], bins=bins, include_lowest=True)
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='range', y='error', data=df)
    plt.xlabel("Rango de Valores Reales")
    plt.ylabel("Error Absoluto")
    plt.title("Error Absoluto por Rango de Valores Reales")
    plt.xticks(rotation=45)
    plt.grid()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Gráfico de error por rango guardado en: {save_path}")