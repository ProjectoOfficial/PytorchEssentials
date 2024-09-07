# PytorchEssentials

## Pytorch Lectures
- Lecture 1: What is a Tensor and how to create it in different ways and on different devices
- Lecture 2: What kind of operations are supported beween Tensors
- Lecture 3: Pytorch automations under the hood. Don't reinvent the wheel!
- Lecture 4: How to optimize parameters through Backpropagation and Gradient Descent
- Lecture 5: Speedup Neural Network creation through torch.nn
- Lecture 6: Manage Vision data thanks to Torchvision
- Lecture 7: Convolutional Neural Networks [1] with Squeeze-and-Excitation [2] Blocks 

## Docker setup
I switched from Anaconda to Docker in order to provide the possibility to run the code on any kind of machine. Futhermore, docker is a tool which can improve the portability and the maintainability of the code within your business, thus this could also be an excuse to learn it.

#### Why docker?
Docker allows you to create something similar to a virtual machine, where any action carried out within it, such as installing an ubuntu package or a python library, is canceled once docker is closed. To make a change made on Docker permanent, just write it in the Dockerfile. For more information check the official repository: https://github.com/ProjectoOfficial/ai-base-docker

In addition to proposing this tool which I think is useful for better managing one's environment, and in any case it is a skill in great demand even within companies (it is worth learning to use it!), I also propose to better organize projects in this way :
- **`project-directory`**: a folder with the name of your project.
  - **`docker`**: import here the *docker* as submodule or in different ways.
  - **`src`**: the main directory where you implement the project.
    - **`requirements`**: the directory where you specify all the requirements files
        - **`base.txt`**: base requirements
        - **`devel.txt`**: development requirements

#### Installation
1. Please follow docker base installation: 
    https://docs.docker.com/engine/install/

    Launch the following commands to install it on Ubuntu:
    ```
    # Add Docker's official GPG key:
    sudo apt-get update
    sudo apt-get install ca-certificates curl
    sudo install -m 0755 -d /etc/apt/keyrings
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
    sudo chmod a+r /etc/apt/keyrings/docker.asc

    # Add the repository to Apt sources:
    echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    ```

    Install docker Packages
    ```
    sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    ```

    Check that docker is correctly installed
    ```
    sudo docker run hello-world
    ```

2. Once docker has been installed, install nvidia-docker2 for GPU support (otherwise you can follow [this](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) procedure (Recommended) ):
    ```
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    ```

    update and install the nvidia container tool
    ```
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    ```

    configure nvidia container tool
    ```
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
    ```

3. Add required permissions to your user in order to perform actions with docker on containers
    ```
    sudo groupadd docker
    sudo usermod -aG docker $USER
    newgrp docker
    ```

4. run docker image:
    ```
    ./run.sh
    ```
    - params:
        - (-d /path/to/dir_to_mount): optionally you can specify supplementary volumes (directory) which tipically can be used as data directory (where you store your datasets). You will find it under ```/home/user/src```
        - (-w): enables the docker to mount a webcam

#### Coding
To be able to program and execute the code inside the docker at the same time (permanent programming, the files will remain even when the docker is closed) I recommend using [VSCode](https://code.visualstudio.com/).

As extensions to do this I use the following. Go to the VSCode marketplace (CTRL+SHIFT+X) and search for:
- ```ms-azuretools.vscode-docker```
- ```ms-vscode-remote.remote-containers```
- ```ms-python.python```

once the extensions have been installed and after launching the docker *run* script, in the menu on the left of VSCode you must select the whale icon (docker), and under the "individual containers" item you will find the container you have just launched with a green arrow next to it. By clicking with the right mouse button on it you will find "attach with VSCode", and this will open a new window for programming inside the docker.

# Bibliography:
- 1: Convolutional Networks for Images, Speech, and
Time-Series [article](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=e26cc4a1c717653f323715d751c8dea7461aa105)
- 2: Squeeze-and-Excitation Networks [article](https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf)

It's not over here, one last step is missing! Go to File>Open Folder -> enter "/home/your_username" as the path
