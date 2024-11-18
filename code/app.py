# app_v3.py

import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
import io

# Configuración del dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cargar el modelo entrenado
#@st.cache_resource
# Definir el modelo ResNet-50 con capas adicionales
class TattooModel(nn.Module):
    def __init__(self, num_classes=6):
        super(TattooModel, self).__init__()
        self.model = models.resnet50(pretrained=True)
        num_features = self.model.fc.in_features

        # Agregar dropout y capas adicionales
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Configuración del dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cargar el modelo entrenado
def load_tattoo_model(model_path, num_classes=6):
    model = TattooModel(num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# Ruta al modelo entrenado
model_path = "C:/Users/theri/Desktop/Trabajos UPC/Ciclo 2024-2/Procesamiento de Imagenes/TF/App/best_tattoo_model_v2.pth"  # Cambiar según la ubicación real
model = load_tattoo_model(model_path)

# Transformaciones para la imagen
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Función para cargar y preprocesar una imagen
def load_and_preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_image = image.copy()
    image_tensor = transform(image)
    return image_tensor, original_image

# Función para realizar la predicción
def predict_tattoo(model, image_bytes, threshold=0.5):
    categories = ['skull', 'dragon', 'knife', 'demon', 'eye', 'other']
    image_tensor, original_image = load_and_preprocess_image(image_bytes)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.sigmoid(outputs).cpu().numpy()[0]

    results = {'predictions': {}, 'detected_categories': []}
    for idx, category in enumerate(categories):
        prob = float(probabilities[idx])
        results['predictions'][category] = {'probability': round(prob * 100, 2), 'detected': prob > threshold}
        if prob > threshold:
            results['detected_categories'].append(category)

    return results, original_image

# Configuración de Streamlit
st.title("Clasificador de Tatuajes Peligrosos")
st.write("Sube una imagen para identificar categorías de tatuajes.")

# Uploader de imágenes
uploaded_image = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Procesar la imagen y realizar predicción
    image_bytes = uploaded_image.read()
    results, original_image = predict_tattoo(model, image_bytes)

    # Mostrar la imagen original
    st.image(original_image, caption="Imagen cargada", use_column_width=True)

    # Mostrar categorías detectadas
    st.write("### Categorías detectadas:")
    if results['detected_categories']:
        st.write(", ".join(results['detected_categories']))
    else:
        st.write("No se detectaron categorías por encima del umbral.")

    # Mostrar probabilidades por categoría
    st.write("### Probabilidades por categoría:")
    for category, info in results['predictions'].items():
        st.write(f"{category}: {info['probability']}%")

    # Mostrar gráfico de barras
    st.write("### Visualización de probabilidades:")
    fig, ax = plt.subplots(figsize=(8, 4))
    categories = list(results['predictions'].keys())
    probabilities = [results['predictions'][cat]['probability'] for cat in categories]
    bars = ax.bar(categories, probabilities, color=["green" if info['detected'] else "gray" for info in results['predictions'].values()])
    ax.axhline(y=50, color='red', linestyle='--', label='Umbral (50%)')
    ax.set_ylabel("Probabilidad (%)")
    ax.set_title("Probabilidades por Categoría")
    ax.legend()
    st.pyplot(fig)