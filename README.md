sudo docker build -t cow .
CUDA_VISIBLE_DEVICES=0,1,2,3 sudo docker run --shm-size 1024m --gpus all cow