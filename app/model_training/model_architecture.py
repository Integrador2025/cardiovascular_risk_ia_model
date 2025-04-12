import tensorflow as tf

def create_regression_model(input_shape):
    inputs = tf.keras.layers.Input(shape=(input_shape,))
    
    x = tf.keras.layers.Dense(
        1024,
        kernel_regularizer=tf.keras.regularizers.L2(1e-3)
    )(inputs)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    x = tf.keras.layers.Dense(
        256,
        kernel_regularizer=tf.keras.regularizers.L2(1e-3)
    )(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    x = tf.keras.layers.Dense(
        128,
        kernel_regularizer=tf.keras.regularizers.L2(1e-3)
    )(x)
    x = tf.keras.layers.LeakyReLU()(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    output = tf.keras.layers.Dense(1, activation='linear')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss='mean_squared_error',
        metrics=['mean_squared_error']
    )

    return model