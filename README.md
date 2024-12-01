# Quick Train

This repository helps you quickly train a deep learning task.

Step1: Get the Docker images.

```bash
docker pull wth1109/quick_train:latest
```

Step2: Run the Docker images .

```bash
docker run -d -v your_code_path:/mnt/code -v your_data_path:/mnt/data -n your_container_name quick_train
```

Step3: Enter Docker container and Activate the virtual environment.

```bash
docker exce -it your_container_name bash
```

```bash
conda activate Quick
```

When you are debugging in Pycharm, you can choose a Docker image for debugging.

Conda location in docker images

```bash
/opt/miniconda/bin/conda
```

Note!!  Deselect the option to use conda for package management.