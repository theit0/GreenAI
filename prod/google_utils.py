from __future__ import annotations
import os
from functools import lru_cache
from typing import Final
import streamlit as st

# ─────────────────────────────────────────────
# Configuración básica
# ─────────────────────────────────────────────
# 1) Clave aquí
API_KEY: Final[str] = st.secrets.get("API_KEY")

# 2) Modelo a utilizar 
MODEL_NAME: Final[str] = "gemini-2.5-flash"        # Cambia a "gemini-2.5-pro" si lo prefieres

# ─────────────────────────────────────────────
# Cliente singleton
# ─────────────────────────────────────────────
@lru_cache(maxsize=1)
def _get_client():
    """Devuelve un cliente de la SDK configurado con la API key."""
    # Importación perezosa para que el módulo se cargue rápido en Streamlit
    from google import genai

    key = os.getenv("API_KEY") or API_KEY
    if not key:
        raise RuntimeError(
            "Falta la API key de Google Gemini. "
        )

    return genai.Client(api_key=key)


# ─────────────────────────────────────────────
# API pública
# ─────────────────────────────────────────────
def get_ai_tip(waste_class: str) -> str:
    """
    Devuelve un tip/estadística breve y aleatoria sobre la clase de residuo indicada.

    Parameters
    ----------
    waste_class : str
        Etiqueta de la clase (por ej. 'plástico', 'glass', 'cartón', etc.).

    Returns
    -------
    str
        Una frase corta (≤ 40 palabras) en español.
    """
    client = _get_client()

    # Prompt en estilo “sistema + usuario” para mayor control
    prompt = (
        "Eres un experto en residuos reciclables. Debes generar una pequeña oración acerca del residuo"
        f"«{waste_class}».  , explicando algun dato curioso, datos del reciclaje del mismo o alguna recomendacion para su desecho.  Debes elegir al azar alguna de las 3 opciones descriptas para generar la oración. No escribas nada mas que la oración por favor."
    )


    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )
    return response.text.strip()


ai_tip = get_ai_tip
