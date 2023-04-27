import numpy as np
import torch
import matplotlib.pyplot as plt

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

from ex1_0_utils import RealNVP, generate_from_model

def interpolate_digits(model, x1, x2):
    
    z1, _ = model.inverse(x1)
    z2, _ = model.inverse(x2)

    fig1, ax1 = plt.subplots(nrows=len(x1), ncols=11)
    fig1.suptitle('Linear interpolation')
    fig1.set_figheight(6)
    fig1.set_figwidth(14)
    
    fig2, ax2 = plt.subplots(nrows=len(x1), ncols=11)
    fig2.suptitle('Sinusoidal interpolation')
    fig2.set_figheight(6)
    fig2.set_figwidth(14)

    for j, lamda in enumerate(np.arange(0, 1+0.1, 0.1)):
        y_lin = model((1 - lamda)*z1 + lamda * z2)  
        y_sin = model((1 - np.sin(lamda*np.pi / 2))*z1 + np.sin(lamda*np.pi / 2) * z2) 
        for i in range(len(x1)):
            ax1[i, j].imshow(y_lin[i].view(28, 28).cpu().detach().numpy(), cmap='binary')
            ax1[i, j].set_axis_off()
            ax2[i, j].imshow(y_sin[i].view(28, 28).cpu().detach().numpy(), cmap='binary')
            ax2[i, j].set_axis_off()
    fig1.savefig('results/ex1_lin_interp.eps')
    fig2.savefig('results/ex1_sin_interp.eps')
    plt.draw()


if __name__ == "__main__":

    torch.manual_seed(42)  # random seed for reproducibility
    torch.set_default_dtype(torch.float32)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    input_dim = 28*28  # input size (MNIST)
    hidden_dim = 256  # output size of a hidden layers
    num_layers = 3  # number of linear layers for each coupling layer

    ## GENERATION

    # 5 coupling layers (trained for 40 epochs)
    model = RealNVP(input_dim=input_dim, num_coupling_layers=5, num_layers=num_layers, device=device).to(device)
    model.eval()
    loaded_state_dict = torch.load("saved_models/RealNVP_5_coupling.pt", map_location=torch.device('cpu'))  # when you train with CUDA but evaluate on CPU
    model.load_state_dict(loaded_state_dict)   

    generate_from_model(model, savepath="results/ex1_gen_5_coupling.eps") 

    # 10 coupling layers (trained for 40 epochs)
    model = RealNVP(input_dim=input_dim, num_coupling_layers=10, num_layers=num_layers, device=device).to(device)
    model.eval()
    loaded_state_dict = torch.load("saved_models/RealNVP_10_coupling.pt", map_location=torch.device('cpu'))  # when you train with CUDA but evaluate on CPU
    model.load_state_dict(loaded_state_dict)

    generate_from_model(model, savepath="results/ex1_gen_10_coupling.eps")

    ## INTERPOLATION

    batch_size = 12

    # Define the dataset and data loader
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)) 
                                ])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    x, _ = next(iter(train_loader))
    x = x.view(-1, 28*28)
    x1 = x[:batch_size // 2]
    x2 = x[batch_size // 2:]

    interpolate_digits(model, x1, x2)

    plt.show()