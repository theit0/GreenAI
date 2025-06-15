import streamlit as st
from PIL import Image
from utils import predict, CLASS_INFO
import pandas as pd

# Ruta a las imágenes (ajusta si tu carpeta es distinta)
ASSETS = {
    0: "prod/assets/cardboard.png", # plástico
    1: "prod/assets/glass.png",     # vidrio
    2: "prod/assets/metal.png",     # metal
    3: "prod/assets/paper.png",     # papel
    4: "prod/assets/plastic.png",   # plástico
    5: "prod/assets/trash.png",     # basura
}


# ────────── Config general ──────────
st.set_page_config(
    page_title="GreenAI – Clasificador de residuos",
    page_icon="♻️",
    layout="centered",
)

st.title("♻️ GreenAI – Clasificador de residuos")

# ────────── Carga de imagen ──────────
uploaded_file = st.file_uploader(
    "Subí una foto del residuo (jpg / png)",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    with st.spinner("Clasificando…"):
        results = predict(img)            # lista ordenada [(idx, prob), …]

    # ── Layout principal ──
    col_img, col_info = st.columns([3, 2])

    with col_img:
        st.image(img, caption="Imagen cargada", use_container_width=True)

    idx, prob = results[0]
    info = CLASS_INFO[idx]

    with col_info:
        # ───────── Cartel de confianza según umbral ─────────
        msg = (f"Categoría: **{info['label'].capitalize()}** "
            f"({prob:.1f} % confianza)")

        if prob >= 85:
            st.success(msg)
        elif prob >= 60:
            st.warning(msg)
        else:
            st.error(msg)

        # ───────── Acción + descripción ─────────
        st.markdown(f"**Acción:** Depositá en el contenedor {info['bin_color']}")
        st.markdown(f"**Descripción:** {info['tip']}")

    # ── Fila horizontal de probabilidades ──
    st.markdown("---")
    st.markdown("#### ¿Qué tan seguro está GreenAI?")

    cols = st.columns(len(results))

    for (cls_idx, p), col in zip(results, cols):
        cls = CLASS_INFO[cls_idx]

        with col:
            # Icono (imagen)
            if cls_idx in ASSETS:
                st.image(ASSETS[cls_idx], width=80)
            else:
                col.markdown(f"<span style='font-size:48px'>{cls['symbol']}</span>",
                            unsafe_allow_html=True)

            # Etiqueta
            st.markdown(f"**{cls['label']}**", unsafe_allow_html=True)

            # Probabilidad
            st.markdown(f"{p:.1f}&nbsp;%", unsafe_allow_html=True)

    # ── Fila horizontal de probabilidades ──
    st.markdown("---")
    st.markdown("#### Tabla + gráfico de probabilidades")

    # ---- Tabla + gráfico de probabilidades ----
    df = pd.DataFrame({
        "Clase": [CLASS_INFO[i]["label"] for i, _ in results],
        "Probabilidad (%)": [p for _, p in results],
    })

    col_table, col_chart = st.columns([1, 1])
    with col_table:
        st.dataframe(df, hide_index=True, use_container_width=True)
    with col_chart:
        st.bar_chart(
            df.set_index("Clase")["Probabilidad (%)"],
            use_container_width=True
        )

    