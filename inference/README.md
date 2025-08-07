# Inference UnwrapNet

This sub-repository contains all the necessary files for fully running the product on `.stl` files, using the same format as the training data. The product is a sleek and intuitive web service for calculating the aerodynamic drag coefficient of 3D car models.

---

## Local Installation:

1. Navigate to the `src` folder.
2. Run the following commands:
```bash
docker build -t aeronet .
docker run --rm -p 8501:8501 --gpus all aeronet
```
3. Open the following URL in your browser: [http://localhost:8501](http://localhost:8501)
4. Wait for the models and dependencies to load.
5. Select a model from the list:
   - ResNet18
   - EfficientNet
   - AutoML
6. Upload your `.stl` file in the provided field.
7. Wait for the result.

---

## NVIDIA Container Toolkit

If the Docker container has issues detecting your GPU, you can most likely resolve them by installing the **NVIDIA Container Toolkit**.

### Steps:
1. Configure the repository:
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
&& sudo apt-get update
```
2. Install the NVIDIA Container Toolkit packages:
```bash
sudo apt-get install -y nvidia-container-toolkit
```
3. Configure the container runtime using `nvidia-ctk`:
```bash
sudo nvidia-ctk runtime configure --runtime=docker
```
4. Restart the Docker daemon:
```bash
sudo systemctl restart docker
