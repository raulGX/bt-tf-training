# bt-tf-training
## Starting tf docker container 
Navigate using the terminal to this repository.

Run the following command:

CPU:
```
docker run -it --name tf -p 8888:8888 -p 6006:6006 -v $(pwd)/notebooks:/notebooks tensorflow/tensorflow:latest-py3
```

GPU:
```
nvidia-docker run -it --name tf -p 8888:8888 -p 6006:6006 -v $(pwd)/notebooks:/notebooks tensorflow/tensorflow:latest-gpu
```

After running it the first time, to restart de container:

```
docker start tf
```

## How to use te Jupyter notebook
1. Copy your data into notebooks/data folder inside this repo (every class should be in separate folders ex: Cats, Dogs, etc.)

2. Open browser after you start the docker container (cpu/gpu) [here](http://localhost:8888/notebooks/train.ipynb#):
3. Optional: run the following commands to start tensorboard inside the container (logs and debugging)
```
sudo docker exec -it tf bash
tensorboard --logdir=/tmp/retrain_logs
```

## Google cloud
1. Create an instance with ubuntu 16
2. Install docker
3. Clone repo
4. Copy data
```
inside ~/bt-tf-training/notebooks
gsutil cp -r gs://streetphotos/data .
```
4. Start docker tf container
5. Log into container
6. Start tensorboard
7. Open Jupyter and start the training process