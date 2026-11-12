import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import gc
import io
import base64
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps, ImageFilter
from flask import Flask, request, jsonify, send_file

# Evitar contención de hilos en el CPU compartido de Render
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

app = Flask(__name__)

# Cargar el modelo sin optimizador y compilar inferencia en grafo estático
model = tf.keras.models.load_model("mi_modelo.h5", compile=False)

@tf.function(reduce_retracing=True)
def inferir(x):
    return model(x, training=False)

# Calentamiento inicial y liberación de memoria temporal
_ = inferir(tf.zeros((1, 28, 28, 1), dtype=tf.float32))
gc.collect()


def calcular_otsu(gray_uint8):
    hist = np.bincount(gray_uint8.ravel(), minlength=256).astype(np.float64)
    total = gray_uint8.size
    w_b = np.cumsum(hist)
    w_f = total - w_b
    valid = (w_b > 0) & (w_f > 0)
    if not np.any(valid):
        return 127

    sum_b = np.cumsum(np.arange(256, dtype=np.float64) * hist)
    sum_total = sum_b[-1]
    m_b = np.zeros(256, dtype=np.float64)
    m_f = np.zeros(256, dtype=np.float64)
    m_b[valid] = sum_b[valid] / w_b[valid]
    m_f[valid] = (sum_total - sum_b[valid]) / w_f[valid]
    var_between = w_b * w_f * ((m_b - m_f) ** 2)
    return int(np.argmax(var_between))


def limpiar_componentes(img_np):
    h, w = img_np.shape
    activo = img_np > 0
    visitado = np.zeros_like(activo, dtype=bool)
    componentes = []

    for y in range(h):
        for x in range(w):
            if activo[y, x] and not visitado[y, x]:
                stack = [(y, x)]
                visitado[y, x] = True
                pixeles = []
                toca_borde = False
                while stack:
                    cy, cx = stack.pop()
                    pixeles.append((cy, cx))
                    if cy == 0 or cy == h - 1 or cx == 0 or cx == w - 1:
                        toca_borde = True
                    for ny in (cy - 1, cy, cy + 1):
                        for nx in (cx - 1, cx, cx + 1):
                            if 0 <= ny < h and 0 <= nx < w:
                                if activo[ny, nx] and not visitado[ny, nx]:
                                    visitado[ny, nx] = True
                                    stack.append((ny, nx))
                componentes.append((pixeles, toca_borde))

    internos = [p for p, borde in componentes if not borde]
    if not internos:
        internos = [p for p, _ in componentes]
    if not internos:
        return img_np

    max_area = max(len(p) for p in internos)
    mascara_limpia = np.zeros_like(img_np)
    for pixeles in internos:
        if len(pixeles) >= max_area * 0.25:
            ys, xs = zip(*pixeles)
            mascara_limpia[ys, xs] = img_np[ys, xs]

    return mascara_limpia


def preprocesar_mnist(image, modo_color="Automático", es_dibujo=False, es_camara=False):
    img_gray = ImageOps.grayscale(image)

    if es_camara:
        w, h = img_gray.size
        lado = int(min(w, h) * 0.60)
        izq = (w - lado) // 2
        sup = (h - lado) // 2
        img_gray = img_gray.crop((izq, sup, izq + lado, sup + lado))

    if not es_dibujo:
        img_gray = img_gray.resize((56, 56), Image.Resampling.BILINEAR)

    img_np = np.array(img_gray, dtype=np.float32)

    if es_dibujo:
        debe_invertir = False
    elif modo_color == "Automático":
        borde = np.concatenate([img_np[0, :], img_np[-1, :], img_np[:, 0], img_np[:, -1]])
        debe_invertir = np.median(borde) > 127
    else:
        debe_invertir = (modo_color == "Fondo claro / número oscuro")

    if debe_invertir:
        img_np = 255.0 - img_np

    if not es_dibujo:
        umbral = calcular_otsu(img_np.astype(np.uint8))
        img_np = np.where(img_np > umbral, img_np, 0.0)
        max_val = img_np.max()
        if max_val > 0:
            img_np = (img_np / max_val) * 255.0
        img_np = limpiar_componentes(img_np)

    img_clean = Image.fromarray(img_np.astype(np.uint8))

    bbox = img_clean.getbbox()
    if bbox is not None:
        img_cropped = img_clean.crop(bbox)
        ancho, alto = img_cropped.size
        if ancho > alto:
            nuevo_ancho = 20
            nuevo_alto = max(1, int(round((alto / ancho) * 20)))
        else:
            nuevo_alto = 20
            nuevo_ancho = max(1, int(round((ancho / alto) * 20)))

        img_20 = img_cropped.resize((nuevo_ancho, nuevo_alto), Image.Resampling.LANCZOS)
        img_28 = Image.new("L", (28, 28), 0)
        pos_x = (28 - nuevo_ancho) // 2
        pos_y = (28 - nuevo_alto) // 2
        img_28.paste(img_20, (pos_x, pos_y))

        img_28 = img_28.filter(ImageFilter.GaussianBlur(radius=0.5))
        arr_28 = np.array(img_28, dtype=np.float32)
        max_28 = arr_28.max()
        if max_28 > 0:
            arr_28 = (arr_28 / max_28) * 255.0
        masa = arr_28.sum()
        if masa > 0:
            ys, xs = np.indices((28, 28))
            cy = (ys * arr_28).sum() / masa
            cx = (xs * arr_28).sum() / masa
            shift_x = int(np.clip(round(14.0 - cx), -4, 4))
            shift_y = int(np.clip(round(14.0 - cy), -4, 4))
            img_centrada = Image.new("L", (28, 28), 0)
            img_centrada.paste(Image.fromarray(arr_28.astype(np.uint8)), (shift_x, shift_y))
            img_28 = img_centrada
    else:
        img_28 = img_clean.resize((28, 28))

    return img_28


@app.route("/")
def index():
    return send_file("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data or "image" not in data:
        return jsonify({"error": "No se recibió ninguna imagen."}), 400

    try:
        _, encoded = data["image"].split(",", 1)
        img_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(img_bytes))
    except Exception:
        return jsonify({"error": "Formato de imagen inválido."}), 400

    es_dibujo = bool(data.get("es_dibujo", False))
    es_camara = bool(data.get("es_camara", False))
    modo_color = data.get("modo_color", "Automático")

    img_28 = preprocesar_mnist(
        image,
        modo_color=modo_color,
        es_dibujo=es_dibujo,
        es_camara=es_camara
    )

    img_array = np.array(img_28, dtype=np.float32) / 255.0

    buf = io.BytesIO()
    img_28.save(buf, format="PNG")
    img_28_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    if np.max(img_array) == 0:
        return jsonify({
            "vacio": True,
            "mensaje": "No se detectó ningún trazo. Dibuja o carga un número primero.",
            "img_28": img_28_b64
        })

    img_input = tf.convert_to_tensor(img_array.reshape(1, 28, 28, 1), dtype=tf.float32)
    probs = inferir(img_input).numpy()[0]

    digito = int(np.argmax(probs))
    confianza = float(np.max(probs)) * 100.0
    segunda_mejor = float(np.partition(probs, -2)[-2]) * 100.0
    margen = confianza - segunda_mejor

    entropia = -float(np.sum(probs * np.log(probs + 1e-10)))
    entropia_norm = entropia / float(np.log(10))

    es_digito_valido = True
    mensaje_error = ""

    if confianza < 90.0:
        es_digito_valido = False
        mensaje_error = "No se detectó un número válido (confianza insuficiente)"
    elif margen < 25.0:
        es_digito_valido = False
        mensaje_error = "Entrada ambigua: no parece un dígito del 0 al 9 (margen bajo)"
    elif entropia_norm > 0.35:
        es_digito_valido = False
        mensaje_error = "Entrada ambigua: patrón no corresponde a un dígito claro"

    return jsonify({
        "vacio": False,
        "es_valido": es_digito_valido,
        "mensaje_error": mensaje_error,
        "digito": digito,
        "confianza": round(confianza, 2),
        "margen": round(margen, 1),
        "probabilidades": [round(float(p) * 100, 2) for p in probs],
        "img_28": img_28_b64
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)