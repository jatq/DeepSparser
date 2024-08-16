## Project description
This project is a python implementation of the our work:`DeepSparser: Denoising Seismic Signal by Integrating Double Sparsity Model and Denoising Autoencoder`, which is submitted and under peer reviewing. Once the paper is published, we will update the readme file.
In this project, we provide the model structure, synthetic and real dataset,config files, as well as training and inference of our method both for real and synthetic data.


## Project structure
- `model/` 
    - `network.py`: define the network structure
    - `model_real.pt`: the trained model on domain data
    - `model_synthetic.pt`: the trained model on synthetic data
- `dataset/`
    - `real/`: real domain data
    - `synthetic/`: synthetic data
    - `dataset_real.py`: load real domain data
    - `dataset_synthetic.py`: load synthetic data
- `config/`
  - `config_real.yaml`: config for real domain data
  - `config_synthetic.yaml`: config for synthetic data
- `main_real.ipynb`: the main file for training and testing the model on real domain data
- `main_synthetic.ipynb`: the main file for training and testing the model on synthetic data
- `download_data.ipynb`: the script to retrive data from online dataset

