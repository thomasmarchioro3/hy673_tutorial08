import numpy as np
import torch
import matplotlib.pyplot as plt

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

from ex3_0_utils import NICER, generate_from_model

if __name__ == "__main__":

    torch.manual_seed(42)  # random seed for reproducibility
    torch.set_default_dtype(torch.float32)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    input_dim = 28*28  # input size (MNIST)
    hidden_dim = 256  # output size of a hidden layers
    num_layers = 3  # number of linear layers for each coupling layer
    prior_type = 'logistic'
    use_scaling = True


    ## GENERATION

    # 10 coupling layers (trained for 40 epochs)
    model = NICER(input_dim=input_dim, num_coupling_layers=10, num_layers=num_layers, device=device, 
                  prior_type=prior_type, use_scaling=use_scaling).to(device)
    model.eval()
    loaded_state_dict = torch.load("saved_models/NICER_10_coupling.pt", map_location=torch.device('cpu'))  # when you train with CUDA but evaluate on CPU
    model.load_state_dict(loaded_state_dict)

    generate_from_model(model, savepath="results/ex3_gen_10_coupling.eps")