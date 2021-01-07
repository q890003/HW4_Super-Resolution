---
Selected-Topics-in-Visual-Recognition-using-Deep-Learning HW4
---
<!-- TOC -->

- [Super Resolution](#super-resolution)
    - [Reproducing the work](#reproducing-the-work)
        - [Enviroment Installation](#enviroment-installation)
        - [Project installation](#project-installation)
    - [Training](#training)

<!-- /TOC -->
# Super Resolution
![](https://i.imgur.com/s2yPhGt.png)


## Reproducing the work
### Enviroment Installation
1. install annoconda
2. create python3.x version 
    ```
    take python3.6 for example
    $ conda create --name (your_env_name) python=3.6
    $ conda activate (your_env_name)
    ```
3. install pytorch 
    - check GPU
        - [Check GPU version](https://www.nvidia.com/Download/index.aspx?lang=cn%20) first and check if CUDA support your GPU.
    - [pytorch](https://pytorch.org/get-started/locally/)
### Project installation
1. clone this repository
    ``` 
    git clone https://github.com/q890003/HW4_Super-Resolution.git
    ```
2. Data
    1. Download Official Image: 
        - [Test data](https://drive.google.com/file/d/1Qv5Jt8_-Uph9KluY_1F6bZi9SYMgzR_R/view?usp=sharing)
        - [Train data](https://drive.google.com/file/d/1AetFdoDA_smxchnXmGsldtQhgB3r18eu/view?usp=sharing)


    2. Put (Test/Train) data to folder, **data/**, under the root dir of this project. 
        ```
        |- HW4_Super-Resolution
            |- data/
                |- Set14/
                    |- (Testing data for validation)
                |- testing_lr_images
                |- training_hr_images
                |- DIV2K.py
                |- common.py
            |- model
                |- architecture.py
                |- block.py
            |- checkpoint_x3/ (Need manual creat)
                |- (step3. parameter_file of model)
            |- results/     (Need manual creat)
            |- README.md
            |- train.py
        ```
    3. Decompress the (Test/Train) data
        ```
        At dir HW4_Super-Resolution/data/
        $ unzip testing_lr_images.zip
        $ unzip training_hr_images.zip
        ```

4. Downoad fine-tuned parameters
    - [IMDN model parameters](https://drive.google.com/file/d/1JsSRb60kXoXlmIq8nODHNsqt9QbFaCyi/view?usp=sharing)
    - put the parameter file to checkpoints folder.
## Training
```
$ python3 train.py
``` 

