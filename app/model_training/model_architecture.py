import tensorflow as tf

def create_multitask_model(input_shape):
    """Crea un modelo de red neuronal multitarea."""
    inputs = tf.keras.layers.Input(shape=(input_shape,))
    
    # Capas compartidas
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
    
    # Rama para clasificación
    clasificacion = tf.keras.layers.Dense(5, activation='softmax', name='clasificacion')(x)
    
    # Rama para regresión
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