import streamlit as st
import requests
from streamlit_js_eval import streamlit_js_eval

try:
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
except Exception:
    st.error(
        "Error al cargar la clave de API de Google. Asegúrate de que esté configurada correctamente en los secretos."
    )
    st.stop()


def get_user_location():
    """Obtiene coordenadas geográficas del usuario vía JS."""
    result = streamlit_js_eval(
        js_expressions="navigator.geolocation.getCurrentPosition",
        key="get_position",
        want_output=True,
    )
    if result and isinstance(result, dict):
        coords = result.get("coords")
        if coords:
            return coords["latitude"], coords["longitude"]
    return None, None


def search_recycling_centers(lat, lng, radius=2000):
    """Busca puntos verdes reales usando Google Places API."""
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "key": GOOGLE_API_KEY,
        "location": f"{lat},{lng}",
        "radius": radius,
        "keyword": "Punto Verde",
        "type": "point_of_interest",
    }
    resp = requests.get(url, params=params)
    data = resp.json()
    if data.get("results"):
        return data["results"]
    return []


def render_route_map(user_lat, user_lng, dest):
    """Muestra un link externo a Google Maps con la ruta."""
    name = dest.get("name", "Punto Verde")
    dest_loc = dest["geometry"]["location"]
    maps_url = f"https://www.google.com/maps/dir/?api=1&origin={user_lat},{user_lng}&destination={dest_loc['lat']},{dest_loc['lng']}&travelmode=walking"

    st.markdown(f"### 🗺️ Ruta al punto verde más cercano: **{name}**")
    st.markdown(
        f"[Haz clic aquí para abrir la ruta en Google Maps 🡕]({maps_url})",
        unsafe_allow_html=True,
    )
