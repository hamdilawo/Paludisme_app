import streamlit as st
import os
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Détection du Paludisme", layout="centered")
st.title("🦟 Détection du Paludisme à partir d'une Image de Frottis")

# Chemin vers le dossier contenant les images
base_path = r"C:\\Users\\lenovo thinkbook\\Documents\\Master1\\Semestre2\\DeepLearning\\Examen_Deep_Learning_Master_Oct2024_DIT (1)\\Examen_Deep_Learning_Master_Oct2023_DIT\\cell_images"

# Choix de la catégorie d'image
category = st.selectbox("Choisir une catégorie d'image :", ["Parasitized", "Uninfected"])

# Lister les fichiers image disponibles
img_dir = os.path.join(base_path, category)
img_files = [f for f in os.listdir(img_dir) if f.endswith(".png")]

# Sélection de l'image
selected_image = st.selectbox("Choisissez une image :", img_files)
img_path = os.path.join(img_dir, selected_image)

# Afficher l'image sélectionnée
img = Image.open(img_path)
st.image(img, caption=f"Image : {selected_image}", use_container_width=True)

# Redimensionner et préparer l'image
img_resized = img.resize((128, 128))
img_array = np.array(img_resized) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Charger le modèle
model_path = "best_CNN.keras"  # Assurez-vous que le fichier est dans le même dossier que ce script
if os.path.exists(model_path):
    model = load_model(model_path)
    prediction = model.predict(img_array)

    # Afficher le résultat
    if prediction[0][0] > 0.5:
        st.success("🦠 Paludisme détecté (cellule infectée)")
    else:
        st.info("✅ Cellule non infectée")
else:
    st.error("Fichier du modèle 'best_CNN.h5' introuvable. Placez-le dans le même dossier que ce script.")