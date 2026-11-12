import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("mi_modelo.h5", compile=False)

pesos = {}
idx = 0
for layer in model.layers:
    cls = layer.__class__.__name__
    if cls in ("Conv2D", "MaxPooling2D", "Flatten", "Dense"):
        w = layer.get_weights()
        pesos[f"type_{idx}"] = np.array(cls)
        if cls == "Conv2D":
            pesos[f"w_{idx}"] = w[0].astype(np.float32)
            pesos[f"b_{idx}"] = w[1].astype(np.float32)
            pesos[f"pad_{idx}"] = np.array(layer.padding)
        elif cls == "MaxPooling2D":
            pesos[f"pool_{idx}"] = np.array(layer.pool_size)
        elif cls == "Dense":
            pesos[f"w_{idx}"] = w[0].astype(np.float32)
            pesos[f"b_{idx}"] = w[1].astype(np.float32)
            act = getattr(layer.activation, "__name__", "linear")
            pesos[f"act_{idx}"] = np.array(act)
        idx += 1

pesos["num_layers"] = np.array(idx)
np.savez_compressed("pesos_modelo.npz", **pesos)
print(f"¡Listo! Pesos exportados correctamente ({idx} capas) -> pesos_modelo.npz")