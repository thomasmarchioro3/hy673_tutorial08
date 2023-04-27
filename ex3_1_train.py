import os
from tqdm import tqdm 

import matplotlib.pyplot as plt

import torch
from torch.optim import Adam

from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

from ex3_0_utils import NICER

if __name__ == "__main__":

    torch.manual_seed(42)  # random seed for reproducibility
    torch.set_default_dtype(torch.float32)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    input_dim = 28*28  # input size (MNIST)
    hidden_dim = 256  # output size of a hidden layers
    num_coupling_layers = 10  # number of coupling layers
    num_layers = 3  # number of linear layers for each coupling layer
    use_scaling = True  # if True adds a constant scaling layer at the end
    prior_type = 'logistic'

    epochs = 40
    batch_size = 128
    lr = 1e-3

    # Define the dataset and data loader
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)) 
                                ])
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        

    # Define the RealNVP model
    model = NICER(input_dim=input_dim, num_coupling_layers=num_coupling_layers, 
                                num_layers=num_layers, 
                                device=device,
                                use_scaling=use_scaling,
                                prior_type='logistic').to(device)

    # Train the model
    model.train()

    # define the optimizer
    optimizer = Adam(model.parameters(), lr=lr)
        
    losses = []
    for epoch in range(epochs):
        tot_log_likelihood = 0
        batch_counter = 0

        for batch_id, (x, _) in enumerate(tqdm(train_loader)):
            
            model.zero_grad()

            x = x.to(device)
            x = x.view(-1, 28*28)  # flatten
            
            z, log_likelihood = model.inverse(x)
            loss = -log_likelihood.mean()  # NLL

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)  # clip the gradient norm for stability
            optimizer.step()      

            tot_log_likelihood -= loss
            losses.append(loss.item())
            batch_counter += 1

        mean_log_likelihood = tot_log_likelihood / batch_counter  # normalize w.r.t. the batches
        print(f'Epoch {epoch+1:d} completed. Log Likelihood: {mean_log_likelihood:.4f}')

    if not os.path.isdir("saved_models"):
        os.makedirs("saved_models")

    if not os.path.isdir("results"):
        os.makedirs("results")

    torch.save(model.state_dict(), f"saved_models/NICER_{num_coupling_layers}_coupling.pt")

    plt.figure()
    plt.plot(torch.arange(1, len(losses)+1, 1), losses)
    plt.xlim([1, len(losses)+1])
    plt.xlabel('Iterations')
    plt.ylabel('Negative Log-likelihood')
    plt.savefig(f'results/ex2_loss_{num_coupling_layers}_coupling.eps')
    plt.show()

