# About
This repository contains the codes and experiments for the paper, Solving Inverse Problems in Snapshot Compressive Imaging with Score Generative Models. Implementation of the score model was forked from [Yang Song's repository](https://github.com/yang-song/score_sde_pytorch) and modified accordingly for our experiments, which was carried out in the environment defined in `score_sci/environment.yml`.

Note that, you may encounter troubles setting up the exact environment and dependencies specified in the configuration file depending on your GPU model and CUDA version. If so, please refer to the original installation guides for relevant packages.

---
# Contact
If you have any doubts or queries, please feel free to reach out at my [email](zhenyuen.dev@gmail.com). Thank you :)

---
# Getting started
## Downloading model checkpoints
1. Download [model checkpoints](https://drive.google.com/drive/folders/1tFmF_uh57O6lx9ggtZT_5LdonVK2cV-e?usp=sharing) provided by Yang Song et al.
2. Store checkpoints under `score_sci/checkpoints/`. E.g.,
    ```
    | score_sci/checkpoints/
    |---- ve/
    |-------- cifar10_ncsnpp_continuous/
    |------------ checkpoint_24.pth
    |-------- ffhq_256_ncsnpp_continuous/
    |------------ checkpoint_48.pth
    ```
3. Setup Conda environment.
    ```
    conda env create --name envname --file=score_sci/environment.yml
    ```

## Viewing the SCI dataset
1. From the dataset is available at [Google Drive](https://drive.google.com/drive/folders/1OAwDAtdy7Nj8ECCUgLEj4AHQD3OoTgWU?usp=sharing), download the `matlab.zip` and `test_gray.zip` under the `SCI` folder.
2. Load the dataset by running the `load_dataset_mat_example.py`.


## Main experiments
1. Navigate to the main folder, `cd ./score_sci`
2. Run demo either through `main.py` or `main.ipynb`
    - Note that, if `main.ipynb` is used, you may have to restart your kernel to clear the GPU memory if CUDA memory limit errors are encountered.
3. Generated samples will be saved under `assets/{scene}_{sampler}` by default

