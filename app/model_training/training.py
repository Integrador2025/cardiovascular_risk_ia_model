import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error
from app.model_training.model_architecture import create_regression_model
from app.model_training.utils import save_model, plot_metrics, save_feature_importances

def train_with_stratified_kfold(X, Y_reg, feature_names, n_splits=5):
    """Entrena el modelo de regresión con validación cruzada estratificada."""
    
    # Crear etiquetas simuladas para estratificación por puntaje (usamos bins)
    bins = np.digitize(Y_reg, bins=[0.0, 0.3, 0.5, 0.7, 0.9, 1.1])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_mse = []

    for fold, (train_index, val_index) in enumerate(skf.split(X, bins), 1):
        print(f"\nEvaluando Fold {fold}/{n_splits}")
        X_train, X_val = X[train_index], X[val_index]
        Y_train, Y_val = Y_reg[train_index], Y_reg[val_index]

        model = create_regression_model(X.shape[1])

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

        model.fit(
            X_train, Y_train,
            validation_data=(X_val, Y_val),
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )


        Y_pred = model.predict(X_val).ravel()
        mse = mean_squared_error(Y_val, Y_pred)
        all_mse.append(mse)
        print(f"MSE fold {fold}: {mse:.4f}")

    # Entrenamiento final
    print("\nEntrenando modelo final con todos los datos...")
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y_reg, test_size=0.2, random_state=42, stratify=bins
    )

    final_model = create_regression_model(X.shape[1])
    history = final_model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=200,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    save_model(final_model, "modelo_regresion_final")

    print("\nResumen MSE en validación cruzada:")
    print(f"MSE promedio: {np.mean(all_mse):.4f} ± {np.std(all_mse):.4f}")

    return final_model, history