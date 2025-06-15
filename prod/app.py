from pathlib import Path
from typing import Tuple

import torch
from PIL import Image
from torchvision import transforms

# ---------- Configuración ----------
MODEL_PATH = Path(__file__).parent / "modelo.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Mapeo índice → etiqueta → instrucciones
CLASS_INFO = {
    0: {
        "label": "plástico",
        "bin_color": "amarillo",
        "symbol": "🔶",
        "tip": "Envases plásticos limpios y secos van al contenedor amarillo."
    },
    1: {
        "label": "papel",
        "bin_color": "azul",
        "symbol": "📘",
        "tip": "Papel y cartón secos y sin restos de comida van al azul."
    },
    2: {
        "label": "metal",
        "bin_color": "amarillo",
        "symbol": "🥫",
        "tip": "Latas y metálicos limpios se depositan en el contenedor amarillo."
    },
    3: {
        "label": "vidrio",
        "bin_color": "verde",
        "symbol": "🍾",
        "tip": "Vidrios sin tapas ni tapones van al contenedor verde."
    },
    4: {
        "label": "cartón",
        "bin_color": "azul",
        "symbol": "📦",
        "tip": "Cartón plegado y sin restos orgánicos va al azul."
    },
    5: {
        "label": "no reciclable",
        "bin_color": "gris",
        "symbol": "🗑️",
        "tip": "Residuos mezclados u orgánicos van al contenedor gris."
    },
}

# ---------- Carga del modelo (cacheada por Streamlit) ----------
@st.cache_resource(show_spinner=False)
def load_model():
    model = torch.load(MODEL_PATH, map_location=DEVICE)
    model.eval()
    return model

# ---------- Preprocesamiento igual al entrenamiento ----------
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # normaliza con las medias/desvs que usaste para entrenar
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ---------- Predicción ----------
def predict(image: Image.Image, topk: int = 1) -> Tuple[int, float]:
    """Devuelve índice de clase y probabilidad top-1 (%)."""
    model = load_model()
    img_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)
        top_prob, top_idx = probs.topk(topk)
    return top_idx.item(), top_prob.item() * 100
