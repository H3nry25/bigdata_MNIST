import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, ImageFilter
import numpy as np
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="CNN MNIST", layout="wide")

@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model('mi_modelo.h5')

model = load_cnn_model()

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

st.title("Reconocimiento de Dígitos MNIST")

def calcular_otsu(gray_uint8):
    hist, _ = np.histogram(gray_uint8.ravel(), bins=256, range=(0, 256))
    total = gray_uint8.size
    sum_total = float(np.dot(np.arange(256), hist))
    sum_b = 0.0
    w_b = 0
    max_var = -1.0
    umbral = 127
    for t in range(256):
        w_b += int(hist[t])
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += float(t * hist[t])
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = float(w_b) * float(w_f) * ((m_b - m_f) ** 2)
        if var_between > max_var:
            max_var = var_between
            umbral = t
    return umbral

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
        img_gray = img_gray.resize((120, 120), Image.Resampling.LANCZOS)

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
        if img_np.max() > 0:
            img_np = (img_np / img_np.max()) * 255.0
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

        # Suavizado ligero y centrado por centro de masa (estándar MNIST)
        img_28 = img_28.filter(ImageFilter.GaussianBlur(radius=0.5))
        arr_28 = np.array(img_28, dtype=np.float32)
        if arr_28.max() > 0:
            arr_28 = (arr_28 / arr_28.max()) * 255.0
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

opcion = st.radio(
    "Selecciona el método de entrada:",
    ["✍️ Pizarra Digital", "📁 Subir Imagen", "📸 Tomar Foto"],
    horizontal=True
)

st.divider()

col_izq, col_der = st.columns([1.1, 1], gap="large")

imagen_para_predecir = None
es_pizarra = False
es_camara = False
modo_fondo = "Automático"

with col_izq:
    if opcion == "✍️ Pizarra Digital":
        st.subheader("Dibuja un dígito (0-9)")
        canvas_result = st_canvas(
            fill_color="#000000",
            stroke_width=28,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=260,
            width=260,
            drawing_mode="freedraw",
            return_image_data=True,
            key=f"canvas_{st.session_state.canvas_key}",
        )

        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            predecir_btn = st.button("🔍 Predecir", type="primary", use_container_width=True)
        with btn_col2:
            if st.button("🗑️ Borrar pizarra", use_container_width=True):
                st.session_state.canvas_key += 1
                st.rerun()

        if predecir_btn and canvas_result.image_data is not None:
            imagen_para_predecir = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
            es_pizarra = True

    elif opcion == "📁 Subir Imagen":
        st.subheader("Sube una imagen")
        uploaded_file = st.file_uploader("Formatos: PNG, JPG, JPEG", type=["png", "jpg", "jpeg"])
        modo_fondo = st.selectbox(
            "Tipo de fondo:",
            ["Automático", "Fondo oscuro / número claro", "Fondo claro / número oscuro"]
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagen original", width=160)
            if st.button("🔍 Predecir Imagen", type="primary", use_container_width=True):
                imagen_para_predecir = image

    elif opcion == "📸 Tomar Foto":
        st.subheader("Captura con cámara")
        st.caption("Centra el número dentro del cuadro de la cámara antes de capturar.")
        camera_file = st.camera_input("Enfoca el número")
        if camera_file is not None:
            image = Image.open(camera_file)
            if st.button("🔍 Predecir Foto", type="primary", use_container_width=True):
                imagen_para_predecir = image
                es_camara = True

with col_der:
    st.subheader("Resultado de la Predicción")
    if imagen_para_predecir is not None:
        img_28 = preprocesar_mnist(
            imagen_para_predecir,
            modo_color=modo_fondo,
            es_dibujo=es_pizarra,
            es_camara=es_camara
        )

        img_array = np.array(img_28, dtype=np.float32) / 255.0

        # 1. Validar que la imagen no esté vacía
        if np.max(img_array) == 0:
            st.warning("⚠️ No se detectó ningún trazo. Dibuja o carga un número primero.")
        else:
            img_array = img_array.reshape(1, 28, 28, 1)
            prediccion = model.predict(img_array)
            digito = int(np.argmax(prediccion))
            confianza = float(np.max(prediccion)) * 100

            # Validación robusta: detectar si NO es un dígito válido
            # Criterios estrictos para rechazar letras/símbolos que parecen dígitos (ej: c≈6, o≈0, etc.)
            probs = prediccion[0]
            segunda_mejor = float(np.partition(probs, -2)[-2]) * 100
            margen = confianza - segunda_mejor

            # Entropía de la distribución: si es alta, el modelo está "confundido"
            entropia = -np.sum(probs * np.log(probs + 1e-10))
            entropia_max = np.log(10)  # entropía máxima para 10 clases uniformes
            entropia_norm = entropia / entropia_max  # 0 = seguro, 1 = totalmente incierto

            es_digito_valido = True
            mensaje_error = ""

            # Umbrales MÁS ESTRICTOS para rechazar letras (c, o, x, etc.)
            if confianza < 90.0:
                es_digito_valido = False
                mensaje_error = "❌ **No se detectó un número válido** (confianza insuficiente)"
            elif margen < 25.0:
                es_digito_valido = False
                mensaje_error = "⚠️ **Entrada ambigua: no parece un dígito del 0 al 9** (margen bajo)"
            elif entropia_norm > 0.35:
                es_digito_valido = False
                mensaje_error = "⚠️ **Entrada ambigua: patrón no corresponde a un dígito claro**"

            res_c1, res_c2 = st.columns([1, 1.5])
            with res_c1:
                st.image(img_28, caption="Entrada (28x28 px)", width=130)
            with res_c2:
                if not es_digito_valido:
                    st.error(mensaje_error)
                    st.caption(f"Confianza máxima: {confianza:.1f}% | Margen vs 2da opción: {margen:.1f}%")
                    st.info("💡 Intenta dibujar el número más claro, centrado y con trazo grueso.")
                else:
                    st.success(f"### Dígito detectado: **{digito}**")
                    st.metric("Confianza", f"{confianza:.2f}%")

            st.write("**Probabilidad por clase (0-9):**")
            st.bar_chart(prediccion[0], height=180)

            # Mostrar detalles de debug si no es válido (opcional, útil para ajustar)
            if not es_digito_valido:
                with st.expander("🔍 Ver detalle de probabilidades (debug)"):
                    for i, p in enumerate(probs):
                        st.write(f"  {i}: {p*100:.2f}%")
    else:
        st.info("Dibuja, sube una imagen o toma una foto a la izquierda y presiona **Predecir** para ver el resultado aquí.")