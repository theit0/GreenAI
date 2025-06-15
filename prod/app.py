import streamlit as st
from PIL import Image
from utils import predict, CLASS_INFO
import pandas as pd

st.set_page_config(page_title="GreenAI – Clasificador de residuos",
                   page_icon="♻️",
                   layout="centered")

st.title("♻️ GreenAI – Clasificador de residuos")

uploaded_file = st.file_uploader(
    "Subí una foto del residuo (jpg / png)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    with st.spinner("Clasificando…"):
        results = predict(img)           

    # Resultado
    col_img, col_info = st.columns([3, 2])   

    with col_img:
        st.image(img, caption="Imagen cargada", use_container_width=True)

    with col_info:
        with st.spinner("Clasificando…"):
            results = predict(img)

        idx, prob = results[0]
        info = CLASS_INFO[idx]

        st.success(f"Categoría: **{info['label'].capitalize()}** "
                   f"({prob:.1f} % confianza)")

        col_sym, col_tip = st.columns([1, 4])
        col_sym.markdown(f"## {info['symbol']}")
        col_tip.markdown(
            f"**Depositá en el contenedor {info['bin_color']}**\n\n"
            f"{info['tip']}"
        )

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