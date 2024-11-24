# Inference Drive

Этот подрепозиторий содержит все файлы необходимые для полноценной работы продукта на .stl файлах такого же формата как и данные для обучения. Продукт представляет собой модный и интуитивный веб сервис для расчета коэффициента аэродинамического сопротивления 3D моделей машин. 
# Локальная установка:
1. Перейдите в папку src
2. Выполните следующие команды.
```
docker build -t aeronet .
docker run -p 8501:8501 --gpus all aeronet
```
3. Выберите модель из списка.
   - ResNet18
   - EffiecientNet
   - AutoML
4. Загрузите файл .stl в соответсвующее поле.

## NVIDIA Container Toolkit
Если при запуске docker-образа будут проблемы с видимостью gpu, то их, скорее всего, можно будет решить с помощью установки NVIDIA Container Toolkit. 
1. Configure the repository:
```shell
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey |sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
&& sudo apt-get update
```
2. Install the NVIDIA Container Toolkit packages:
```shell
sudo apt-get install -y nvidia-container-toolkit
```
3. Configure the container runtime by using the nvidia-ctk command:
```shell
sudo nvidia-ctk runtime configure --runtime=docker
```
4.Restart the Docker daemon:
```shell
sudo systemctl restart docker
```
   
