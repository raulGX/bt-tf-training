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