# Train UnwrapNet

## Launch
This sub-repository contains all files necessary to reproduce our experiments. If you'd like to verify our results, simply run the following commands:

```shell
sudo docker build -t drive .
CUDA_VISIBLE_DEVICES=0 sudo docker run --gpus all drive
```

## Contents
The repository contains code for training **ResNet18**, **EfficientNet**, and an ensemble model based on their outputs. 

We start training from pretrained checkpoints, modifying the input layer to accept **9-channel images** (concatenation of 5 different vehicle projections). 

If you'd like to change training hyperparameters, you can do so in the `config.yaml` file.

## RuntimeError: DataLoader worker (pid 117) is killed by signal: Bus error.
To fix this error, increase the shared memory size using `--shm-size=1024m`:

```shell
CUDA_VISIBLE_DEVICES=0 sudo docker run --shm-size=1024m --gpus all drive
```

If the error persists, try increasing the shared memory further, e.g., `--shm-size=2048m`.

## NVIDIA Container Toolkit
If the Docker container doesn't detect the GPU, the issue can likely be resolved by installing the **NVIDIA Container Toolkit**.

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

3. Configure the container runtime:
```shell
sudo nvidia-ctk runtime configure --runtime=docker
```

4. Restart the Docker daemon:
```shell
sudo systemctl restart docker
