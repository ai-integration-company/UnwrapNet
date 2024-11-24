# Drive

## Запуск
Этот репозиторий содержит все файлы необходимые для воспроизведения наших экспериментов. Если вы хотите просто проверить наши результаты, то можно просто выполнить следующую последовательность команд:
```shell
sudo docker build -t drive .
CUDA_VISIBLE_DEVICES=0 sudo docker run --gpus all drive
```
## Содержимое
В репозитории содержится код для обучения resnet18 и efficientnet и ансамбля на результатах их работы. Мы начинаем наше обучение с уже натренированных чекпоинтов, у которых меняем входной слой, чтобы он принимал 9-ти канальное изображение (конкатенация 5-ти разных проекций автомобиля). Если вы хотите поменять какие-то гиперпараметры связанные с обучением, это можно сделать с помощью config.yaml.
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
