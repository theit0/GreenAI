from pathlib import Path
from typing import Tuple
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import torchvision.models as models

# ---------- Configuración ----------
MODEL_PATH = Path(__file__).parent / "modelo.pth"
print(MODEL_PATH)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Índice → información de la clase
CLASS_INFO = {
    0: {
        "label": "cartón",
        "bin_color": "azul",
        "symbol": "📦",
        "tip": "Dobla el cartón para que ocupe menos espacio y colócalo limpio y seco en el contenedor azul."
    },
    1: {
        "label": "vidrio",
        "bin_color": "verde",
        "symbol": "🍾",
        "tip": "Retira tapas o corchos y deposita el vidrio en el contenedor verde."
    },
    2: {
        "label": "metal",
        "bin_color": "amarillo",
        "symbol": "🥫",
        "tip": "Enjuaga latas u otros metales ligeros y llévalos al contenedor amarillo."
    },
    3: {
        "label": "papel",
        "bin_color": "azul",
        "symbol": "📄",
        "tip": "Papel limpio, sin grapas ni restos de comida, va en el contenedor azul."
    },
    4: {
        "label": "plástico",
        "bin_color": "amarillo",
        "symbol": "🧴",
        "tip": "Envases plásticos limpios y secos se desechan en el contenedor amarillo."
    },
    5: {
        "label": "no reciclable",
        "bin_color": "gris",
        "symbol": "🗑️",
        "tip": "Residuos mezclados u orgánicos se depositan en el contenedor gris."
    },
}

# ---------- Construcción de la arquitectura ----------
def build_model(num_classes: int = 6):
    """Devuelve la arquitectura exacta usada en entrenamiento."""
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model

# ---------- Carga del modelo (cacheada por Streamlit) ----------
@st.cache_resource(show_spinner=False)
def load_model():
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model = build_model()
    model.load_state_dict(state_dict) # aquí cargamos pesos
    model.to(DEVICE)
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
def predict(image: Image.Image, topk: int | None = None):
    """
    Devuelve una lista ordenada de (idx, prob) para todas las clases
    o solo las `topk` primeras si se indica.
    Prob se da en porcentaje (0-100).
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    model = load_model()
    img_tensor = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

    if topk is None:
        topk = probs.size(0)

    top_prob, top_idx = probs.topk(topk)          # ya vienen ordenadas
    return [(idx.item(), prob.item() * 100) for idx, prob in zip(top_idx, top_prob)]