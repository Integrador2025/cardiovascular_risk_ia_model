import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def save_model(model, model_name="modelo_entrenado"):
    """Guarda el modelo en disco usando el formato .keras recomendado."""
    model_dir = "model"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}.keras")  # Usamos extensión .keras
    model.save(model_path)  # Eliminamos el argumento save_format
    print(f"Modelo guardado en formato Keras moderno en: {model_path}")

def plot_metrics(history, metric, save_path, branch='clasificacion'):
    """Genera y guarda gráficos de métricas de entrenamiento."""
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