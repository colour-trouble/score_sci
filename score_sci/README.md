## Getting started
### Downloading model checkpoints
1. Download [model checkpoints](https://drive.google.com/drive/folders/1tFmF_uh57O6lx9ggtZT_5LdonVK2cV-e?usp=sharing) provided by Yang Song et al.
2. Store checkpoints under `assets/`. E.g.,
    ```
    | ./assets/
    |---- ve/
    |-------- cifar10_ncsnpp_continuous/
    |------------ checkpoint_24.pth
    |-------- ffhq_256_ncsnpp_continuous/
    |------------ checkpoint_48.pth
    ```
3. Setup Conda environment.
    ```
    conda env create --name envname --file=environment.yml
    ```

## Usage
1. Run demo either through `main.py` or `main.ipynb`
    - Note that, if `main.ipynb` is used, you may have to restart your kernel to clear the GPU memory if CUDA memory limit errors are encountered. 