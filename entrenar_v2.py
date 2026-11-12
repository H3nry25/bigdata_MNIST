import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Cargar y normalizar MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# 2. Arquitectura CNN mejorada con Data Augmentation (diapositivas 28-30)
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10),
    layers.RandomTranslation(0.08, 0.08),

    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 3. Medir tiempo de entrenamiento (model.fit)
inicio_train = time.time()
model.fit(x_train, y_train, epochs=8, batch_size=128, validation_data=(x_test, y_test))
tiempo_train = time.time() - inicio_train

# 4. Guardar modelo ganador y extraer métricas de la diapositiva 22
ruta_modelo = "mi_modelo.h5"
model.save(ruta_modelo)

_, test_acc = model.evaluate(x_test, y_test, verbose=0)
params_entrenables = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))
tamano_mb = os.path.getsize(ruta_modelo) / (1024 * 1024)

# Medir tiempo de inferencia promedio en 1 imagen
muestra = x_test[0:1]
_ = model.predict(muestra, verbose=0)  # calentamiento
inicio_inf = time.time()
for _ in range(50):
    _ = model.predict(muestra, verbose=0)
tiempo_inf_ms = ((time.time() - inicio_inf) / 50) * 1000

print("\n--- MÉTRICAS MODELO 2 (PARA TABLA COMPARATIVA) ---")
print(f"Accuracy (%): {test_acc * 100:.2f}%")
print(f"# Parámetros entrenables: {params_entrenables:,}")
print(f"Tamaño del modelo (MB): {tamano_mb:.2f} MB")
print(f"Tiempo de entrenamiento: {tiempo_train:.2f} segundos")
print(f"Tiempo de inferencia: {tiempo_inf_ms:.2f} ms")