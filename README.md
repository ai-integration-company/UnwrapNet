# UnwrapNet

Code for paper UnwrapNet: A Novel Approach To Proceed 3D Objects.

This repository contains all files necessary to reproduce our experiments and run the product.  
If you want to **reproduce the experiment**, go to the **train** folder and follow the instructions.  
If you want to **run the product with trained models**, go to the **inference** folder and follow the instructions.

---

## Solution Description

We use 3D car models to generate 5 images:

1. **Three grayscale projections** from different sides, where each pixel value represents the distance to the projection plane.  
2. **Spherical and cylindrical RGB unwrapped projections** of the model.

All computationally expensive operations are implemented in multithreaded **C++ code**, enabling extremely fast image generation from 3D models (1000 models per minute).

Examples of generated images:

### Grayscale Projections
<div style="display: flex; justify-content: center;">
  <img src="img/up_projection.png" alt="up Projection" width="20%">
  <img src="img/left_projection.png" alt="left Projection" width="20%">
  <img src="img/front_projection.png" alt="front Projection" width="20%">
</div>

### RGB Unwrapped Projections
<div style="display: flex; justify-content: center;">
  <img src="img/spherical_projection.png" alt="Spherical Projection" width="20%">
  <img src="img/cylinder_projection.png" alt="Cylindrical Projection" width="20%">
</div>

---

We trained two pretrained models on this data, modifying:
- the first layer to accept **9 channels** as input (3 grayscale + 3 RGB spherical + 3 RGB cylindrical)
- the final layer to output a single value:

- **ResNet18**
- **EfficientNetB1**

Both models are used to predict the aerodynamic drag coefficient.  
We then trained a lightweight ensemble model on the outputs of these two networks.  
This approach provides a **flexible trade-off between speed and accuracy**.

- Use **EfficientNet** for **speed**.  
- Use the full pipeline, which we named **UnwrapNet**, for **accuracy**.

## Data Preprocessing

In the `train` subdirectory, preprocessed data (5 images per model) are downloaded automatically.  
To generate training/inference-ready data from 3D models manually:

1. Download `.stl` models into the `stl_dir` directory.
2. If a model contains more than **100,000 points**, randomly sample **100,000** points.
3. Install the required dependencies for compiling the C++ code:
```bash
apt-get update && apt-get install -y \
    build-essential \
    g++ \
    libopencv-dev \
    libeigen3-dev \
    libassimp-dev \
    cmake \
    && apt-get clean
```
4. Download and compile the file `inference/src/mesh_projection_mt.cpp`:
```bash
g++ -std=c++17 -O2 -o mesh_projection_mt mesh_projection_mt.cpp \
    -I/usr/include/opencv4 \
    -I/usr/include/eigen3 \
    -lopencv_core \
    -lopencv_imgcodecs \
    -lopencv_highgui \
    -lopencv_imgproc \
    -lassimp \
    -pthread
```
5. Create the `img_dir` directory and give full permissions:
```bash
mkdir img_dir
chmod 777 img_dir
```
6. Run the compiled file with the necessary arguments:
```bash
./mesh_projection_mt stl_dir img_dir 3
