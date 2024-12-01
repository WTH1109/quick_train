# Quick Train

This repository helps you quickly train a deep learning task.

**Step1: Get the Docker images.**

```bash
docker pull wth1109/quick_train:latest
```

**Step2: Run the Docker images .**

```bash
docker run -id -v your_code_path:/mnt/code -v your_data_path:/mnt/data -n your_container_name wth1109/quick_train
```

**Step3: Enter Docker container and Activate the virtual environment.**

```bash
docker exce -it your_container_name bash
```

```bash
conda activate Quick
```

**Step4: Setting the Pycharm IDE** 

When you want  to debug in Pycharm, you can choose a Docker image for debugging.  

The specific steps are as follows:

File->Setting->Python Interpreter->Add Interpreter->On Docker->Use exisiting->wth1109/quick_train->Conda Environment

Conda location in docker images

```bash
/opt/miniconda/bin/conda
```

If PyCharm cannot detect the package, try the following steps.

File->Setting->Python Interpreter->Show All->Show Interpreter Path->Add:

```bash
/.conda/envs/Quick/lib/python3.9/site-packages
```

Then clear Pycharm cache:

File->Invalidate Caches->Clear file system cache and Local History & Mark download shared indexes as broken->invalidation and Restart