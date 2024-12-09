import os
import glob
import logging
import shutil

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms

import joblib
import pickle
import kagglehub

import streamlit as st
import subprocess

from stl import mesh

from models import EfficientNetRegression

logger = logging.getLogger("MyLogger")
logger.setLevel(logging.DEBUG)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&display=swap'); /* Replace with your desired Cyberpunk font */
    
    .stApp {
        background-color: rgb(0, 0, 0); 
    }
    h1, h2, h3, h4, h5, h6, p, span, label {
        font-family: 'Orbitron', sans-serif; /* Use the custom font */
        color: rgb(60, 255, 2); /* Neon green for Cyberpunk style */
    }
    h1 {
        color: rgb(60, 255, 2);
        font-size: 3em; /* Adjust size for the main title */
        text-shadow: 0 0 5px rgb(60, 255, 2), 0 0 10px rgb(60, 255, 2), 0 0 20px rgb(60, 255, 2);
    }
    h2 {
        color: rgb(255, 35, 166); /* Neon pink for subtitles */
        font-size: 2em;
        text-shadow: 0 0 5px rgb(255, 35, 166), 0 0 10px rgb(255, 35, 166), 0 0 20px rgb(255, 35, 166);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1>AERO DRIVE</h1>", unsafe_allow_html=True)
st.markdown("<h2>AI Integration</h2>", unsafe_allow_html=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def remove_module_prefix(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[len('module.'):]
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict


def load_EfficientNet():
    # Указываем путь к модели внутри контейнера
    model_path = "model/efficientNet.pth"

    # Загружаем чекпоинт модели
    checkpoint = torch.load(model_path)

    # Инициализируем модель
    model = EfficientNetRegression()
    model.load_state_dict(checkpoint)

    # Переносим модель на устройство (GPU/CPU)
    return model.to(device)


def load_ResNet():
    # Указываем путь к модели внутри контейнера
    model_path = "model/ResNet.pth"

    # Загружаем чекпоинт модели
    checkpoint = torch.load(model_path)

    model = models.resnet18(pretrained=True)

    original_conv = model.conv1

    # Новый вход с 9 каналами
    new_conv = nn.Conv2d(9, original_conv.out_channels, kernel_size=original_conv.kernel_size,
                         stride=original_conv.stride, padding=original_conv.padding, bias=original_conv.bias)

    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = original_conv.weight
        new_conv.weight[:, 3:6, :, :] = original_conv.weight
        new_conv.weight[:, 6:9, :, :] = original_conv.weight
    model.conv1 = new_conv

    # Слой регрессии
    model.fc = nn.Linear(model.fc.in_features, 1)

    model.load_state_dict(checkpoint)
    return model.to(device)


def load_autoML():
    # Путь к модели внутри контейнера
    automl_model_path = "model/automl_model_norm.pkl"

    # Загружаем модель
    automl_model = joblib.load(automl_model_path)
    return automl_model


def run_inference(model, dataloader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for images in dataloader:
            images = images.unsqueeze(0).to(device)

            outputs = model(images)
            predictions.extend(outputs.cpu().numpy().flatten())
    return predictions


def sample_points(points, n):
    n = min(len(points), n)
    indices = np.random.choice(len(points), size=n, replace=False)
    sampled_points = points[indices]
    return sampled_points


def save_sampled_stl(points, file_path):
    degenerate_triangles = np.array([[p, p, p] for p in points])
    new_mesh = mesh.Mesh(np.zeros(degenerate_triangles.shape[0], dtype=mesh.Mesh.dtype))
    for i, triangle in enumerate(degenerate_triangles):
        new_mesh.v0[i] = triangle[0]
        new_mesh.v1[i] = triangle[1]
        new_mesh.v2[i] = triangle[2]
    new_mesh.save(file_path)


@st.cache_resource(show_spinner=False)
def handle_stl_file(stl_file):
    """Handles saving the STL file, generating projections, and preparing paths."""
    input_dir = 'uploaded_files'
    output_dir = 'images'

    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(input_dir, stl_file.name)

    with open(file_path, "wb") as f:
        f.write(stl_file.getbuffer())

    stl_mesh = mesh.Mesh.from_file(file_path)

    total_vertices = len(stl_mesh.v0) + len(stl_mesh.v1) + len(stl_mesh.v2)
    if total_vertices > 1_000_000:
        st.warning(f"Large STL file detected with {total_vertices} vertices. Sampling 200,000 points.")
        points = np.vstack([stl_mesh.v0, stl_mesh.v1, stl_mesh.v2])
        sampled_points = sample_points(points, 200_000)
        sampled_file_path = os.path.join(input_dir, f"sampled_{stl_file.name}")
        save_sampled_stl(sampled_points, sampled_file_path)
        st.success(f"Sampled STL file saved at {sampled_file_path}")
        file_path = sampled_file_path
    else:
        stl_mesh.save(file_path)

    subprocess.run(["./mesh_projection_mt", input_dir, output_dir, '3'], capture_output=True)
    st.success('Projections generated!')

    return os.path.join(output_dir, stl_file.name[:-4])


@st.cache_resource(show_spinner=False)
def load_images(img_path):
    """Loads and categorizes images into grayscale and RGB."""
    grayscale_images = []
    rgb_images = []

    for img in glob.glob(img_path + '/*.png'):
        if 'spherical' in img or 'cylinder' in img:
            rgb_images.append(Image.open(img).convert('RGB'))
        elif 'up' in img or 'left' in img or 'front' in img:
            grayscale_images.append(Image.open(img).convert('L'))

    return grayscale_images, rgb_images


@st.cache_resource(show_spinner=False)
def preprocess_images(grayscale_images, rgb_images):
    """Preprocesses images and returns the concatenated tensor."""
    image_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    images = [image_transforms(img) for img in grayscale_images + rgb_images]
    return torch.cat(images, dim=0)


@st.cache_resource(show_spinner=False)
def get_img():
    img_path = 'images'
    img_path = os.path.join(img_path, stl_file.name[:-4])

    # Порядок слоев
    ordered_files = [
        "spherical_projection.png",
        "cylinder_projection.png",
        "left_projection.png",
        "front_projection.png",
        "up_projection.png",
    ]

    images = []
    img_paths = []

    for file_name in ordered_files:
        file_path = os.path.join(img_path, file_name)
        if os.path.exists(file_path):
            if "spherical" in file_name or "cylinder" in file_name:
                images.append(Image.open(file_path).convert('RGB'))
            elif "up" in file_name or "left" in file_name or "front" in file_name:
                images.append(Image.open(file_path).convert('L'))
            img_paths.append(file_path)

    return images, img_paths


@st.cache_resource(show_spinner=False)
def load_models():
    """Loads models and caches them."""
    model_efficientnet = load_EfficientNet()
    model_resnet = load_ResNet()
    model_autoML = load_autoML()
    return model_efficientnet, model_resnet, model_autoML


with st.spinner('Загрузка моделей, подождите...'):
    model_efficientnet, model_resnet, model_autoML = load_models()

using_model = st.radio('Выберите модель: ', ('ResNet18', 'EffiecientNet', 'AERONET'))

# Загрузка stl файла
st.subheader("Загрузите stl файл")

# Загружаем STL файл
stl_file = st.file_uploader("")

if "uploaded_stl" not in st.session_state:
    st.session_state["uploaded_stl"] = None

if stl_file and stl_file != st.session_state["uploaded_stl"]:
    st.session_state["uploaded_stl"] = stl_file
    st.cache_resource.clear()

if stl_file is not None:
    with st.spinner('Проецируем машину на сферу...'):
        img_path = handle_stl_file(stl_file)

    grayscale_images, rgb_images = load_images(img_path)

    columns = st.columns(3)

    for i, img in enumerate(grayscale_images[:3]):
        with columns[i]:
            st.image(img, use_container_width=True)

    columns = st.columns(2)
    for i, img in enumerate(rgb_images[:2]):
        with columns[i]:
            st.image(img, use_container_width=True)

    images, img_paths = get_img()

    # st.write(f"{img_paths}")

    image_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    images = [image_transforms(img) for img in images]

    input_tensor = torch.cat(images, dim=0)

    outputs = 0.0
    with torch.no_grad():
        if using_model == 'ResNet18':
            model_resnet.eval()
            input_tensor = input_tensor.unsqueeze(0).to(device)
            outputs = model_resnet(input_tensor).squeeze().item()
        elif using_model == 'EffiecientNet':
            model_efficientnet.eval()
            input_tensor = input_tensor.unsqueeze(0).to(device)
            outputs = model_efficientnet(input_tensor).squeeze().item()
        elif using_model == 'AERONET':
            model_resnet.eval()
            model_efficientnet.eval()

            input_tensor = input_tensor.unsqueeze(0).to(device)

            outputs_resnet = model_resnet(input_tensor).squeeze().item()
            outputs_efficientNet = model_efficientnet(input_tensor).squeeze().item()
            nn_outputs = pd.DataFrame({
                'resnet18_Prediction': [outputs_resnet],
                'effnetb1full_Prediction': [outputs_efficientNet]
            })
            outputs = model_autoML.predict(nn_outputs).data.squeeze().item()

    st.markdown(
        f"""
        <div style="
            border: 3px solid rgb(60, 255, 2); 
            border-radius: 10px; 
            background-color: rgb(0, 0, 0); 
            padding: 20px; 
            margin: 20px 0; 
            text-align: center;">
            <h1 style="color: rgb(60, 255, 2); font-size: 2.5em;">Аэродинамическое сопротивление:</h1>
            <h2 style="color: rgb(255, 166, 35); font-size: 2em;">{outputs:.5f}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    # st.metric(label="Результат", value=f"{outputs:.2f}", delta=None)

    # # Удалим файлы из папок:
    # for folder_path in ['uploaded_files', 'images']:
    #     # Убедитесь, что папка существует
    #     if os.path.exists(folder_path):
    #         # Перебираем файлы в папке
    #         for file_name in os.listdir(folder_path):
    #             file_path = os.path.join(folder_path, file_name)

    #             # Проверяем, что это файл (не папка)
    #             if os.path.isfile(file_path):
    #                 os.remove(file_path)  # Удаляем файл

    # Удалим файлы из папок:
    for dir_path in ['uploaded_files', 'images']:
        # Удаляем все содержимое папки images
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)  # Удаление файла
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Удаление папки и её содержимого
