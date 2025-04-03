import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score,
    mean_squared_error
)
from model_architecture import create_multitask_model
from utils import save_model, plot_metrics, save_feature_importances

def train_with_stratified_kfold(X, Y, feature_names, n_splits=20):
    """Entrena el modelo con validación cruzada pero guarda solo un modelo final"""
    Y_class, Y_reg = Y  # Separar objetivos
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Listas para almacenar métricas de todos los folds
    all_metrics = {
        'accuracy': [],
        'auc_roc': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'mse': []
    }
    
    # 1. Fase de evaluación con validación cruzada
    for fold, (train_index, val_index) in enumerate(skf.split(X, Y_class), 1):
        print(f"\nEvaluando Fold {fold}/{n_splits}")
        X_train, X_val = X[train_index], X[val_index]
        Y_train_class, Y_train_reg = Y_class[train_index], Y_reg[train_index]
        Y_val_class, Y_val_reg = Y_class[val_index], Y_reg[val_index]
        
        # Modelo temporal para evaluación
        temp_model = create_multitask_model(X_train.shape[1])
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_clasificacion_loss', 
            patience=15, 
            restore_best_weights=True,
            mode='min'
        )
        
        temp_model.fit(
            X_train, 
            {'clasificacion': Y_train_class, 'regresion': Y_train_reg},
            validation_data=(X_val, {'clasificacion': Y_val_class, 'regresion': Y_val_reg}),
            epochs=200,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluación y almacenamiento de métricas
        preds = temp_model.predict(X_val)
        Y_pred_proba = preds[0]
        Y_pred_class = np.argmax(Y_pred_proba, axis=1)
        Y_pred_reg = preds[1].ravel()
        
        # Almacenar métricas
        all_metrics['accuracy'].append(precision_score(Y_val_class, Y_pred_class, average='weighted', zero_division=0))
        all_metrics['precision'].append(precision_score(Y_val_class, Y_pred_class, average='weighted', zero_division=0))
        all_metrics['recall'].append(recall_score(Y_val_class, Y_pred_class, average='weighted', zero_division=0))
        all_metrics['f1'].append(f1_score(Y_val_class, Y_pred_class, average='weighted', zero_division=0))
        all_metrics['mse'].append(mean_squared_error(Y_val_reg, Y_pred_reg))
        
        try:
            auc_roc = roc_auc_score(Y_val_class, Y_pred_proba, multi_class='ovr')
            all_metrics['auc_roc'].append(auc_roc)
        except Exception as e:
            print(f"No se pudo calcular AUC-ROC: {e}")
    
    # 2. Fase de entrenamiento final con todos los datos
    print("\nEntrenando modelo final con todos los datos...")
    final_model = create_multitask_model(X.shape[1])
    
    # Usamos el 20% de los datos para validación final
    from sklearn.model_selection import train_test_split
    X_train, X_val, Y_train_class, Y_val_class = train_test_split(
        X, Y_class, test_size=0.2, stratify=Y_class, random_state=42
    )
    _, _, Y_train_reg, Y_val_reg = train_test_split(
        X, Y_reg, test_size=0.2, random_state=42
    )
    
    history = final_model.fit(
        X_train,
        {'clasificacion': Y_train_class, 'regresion': Y_train_reg},
        validation_data=(X_val, {'clasificacion': Y_val_class, 'regresion': Y_val_reg}),
        epochs=200,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Guardar el modelo único final
    save_model(final_model, "modelo_final")
    
    # Guardar métricas de validación cruzada
    print("\nResumen de métricas en validación cruzada:")
    for metric, values in all_metrics.items():
        print(f"{metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")
    
    # Guardar gráficos y características importantes
    plot_metrics(history, "loss", "model/loss_final.png", branch='clasificacion')
    plot_metrics(history, "accuracy", "model/accuracy_final.png", branch='clasificacion')
    save_feature_importances(final_model, feature_names, "model/feature_importances_final.csv")
    
    return final_model