import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Uniform, Distribution, Normal


"""
RealNVP coupling layer
"""
class CouplingLayer(nn.Module):

    def __init__(self, input_dim, output_dim=None, hidden_dim=256, mask=None, num_layers=4):
        super().__init__()

        self.input_dim = input_dim
        if output_dim is None:
            self.output_dim = input_dim
        else:
            self.output_dim = output_dim

        self.hidden_dim = hidden_dim

        if mask is None:
            raise ValueError("Error: Mask cannot be empty")

        self.mask = mask
        

        modules = [nn.Linear(input_dim, hidden_dim), 
                    nn.LeakyReLU(0.2)]

        for _ in range(num_layers - 2):
            modules.append(nn.Linear(hidden_dim, hidden_dim))
            modules.append(nn.LeakyReLU(0.2))
        modules.append(nn.Linear(hidden_dim, 2*self.output_dim))

        self.ms = nn.Sequential(*modules)

    def forward(self, z):
        z_A = self.mask * z
        z_B = (1 - self.mask) * z
        x_A = z_A

        ms_z_A = self.ms(z_A)

        # use half of the output for the additive term and half for scaling
        m_z_A = ms_z_A[:, :self.input_dim]  
        alpha_z_A = torch.tanh(ms_z_A[:, self.input_dim:])  # tanh bounds alpha in [-1, 1]
        s_z_A = torch.exp(alpha_z_A)  # scaling factor bounded in [1/e, e]

        x_B = s_z_A * z_B + (m_z_A * (1 - self.mask))
        return x_A + x_B
    
    # inverse mapping
    def inverse(self, x):
        x_A = self.mask * x
        x_B =(1 - self.mask) * x
        z_A = x_A

        ms_z_A = self.ms(z_A)
        m_z_A = ms_z_A[:, :self.input_dim]
        alpha_z_A = torch.tanh(ms_z_A[:, self.input_dim:])
        s_z_A = torch.exp(-alpha_z_A)
        log_det_jacobian = torch.sum(-alpha_z_A, dim=-1)

        z_B = s_z_A * (x_B - m_z_A * (1 - self.mask))
        return z_A + z_B, log_det_jacobian
    

"""
Logistic Distribution (from tutorial 4)
"""
class LogisticDistribution(Distribution):
  def __init__(self):
    super().__init__()

  def log_prob(self, x):
    return -(F.softplus(x) + F.softplus(-x))

  def sample(self, size):
    z = Uniform(torch.FloatTensor([0.]), torch.FloatTensor([1.])).sample(size)

    return torch.log(z) - torch.log(1. - z)

"""
RealNVP
"""
class RealNVP(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_coupling_layers=3, num_layers=6, device='cpu', 
                 prior_type='normal'):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_coupling_layers = num_coupling_layers
        self.num_layers = num_layers  # number of linear layers for each coupling layer
        self.prior_type = prior_type

        # alternating mask orientations for consecutive coupling layers
        masks = [self._get_mask(input_dim, orientation=(i % 2 == 0)).to(device)
                                                for i in range(num_coupling_layers)]

        self.coupling_layers = nn.ModuleList([CouplingLayer(input_dim=input_dim,
                                    hidden_dim=hidden_dim,
                                    mask=masks[i], num_layers=num_layers)
                                for i in range(num_coupling_layers)])

        if prior_type == 'logistic':
            self.prior = LogisticDistribution()
        elif prior_type == 'normal':
            self.prior = Normal(0, 1)
        else:
            print("Error: Invalid prior_type")
        
        self.device = device

    def forward(self, z):
        x = z
        for i in range(len(self.coupling_layers)):  # pass through each coupling layer
            x = self.coupling_layers[i](x)
        return x

    def inverse(self, x):
        z = x

        # initialize log-likelihood
        log_likelihood = 0

        for i in reversed(range(len(self.coupling_layers))):  # pass through each coupling layer in reversed order
            z, log_det_jacobian = self.coupling_layers[i].inverse(z)
            log_likelihood += log_det_jacobian  # add the log_det_jacobian of each layer

        log_likelihood += torch.sum(self.prior.log_prob(z), dim=-1)

        return z, log_likelihood

    def sample(self, num_samples):
        z = self.prior.sample([num_samples, self.input_dim]).view(num_samples, self.input_dim)
        z = z.to(self.device)
        return self.forward(z)

    def _get_mask(self, dim, orientation=True):
        mask = torch.zeros(dim)
        mask[::2] = 1.
        if orientation:
            mask = 1. - mask  # flip mask if orientation is True
        return mask.float()
    

def generate_from_model(model, savepath=None):
    xhat = model.sample(8)

    fig, ax = plt.subplots(nrows=2, ncols=4)
    fig.set_figheight(6)
    fig.set_figwidth(12)
    i = 0
    for row in ax:
        for col in row:
            col.imshow(xhat[i].cpu().detach().numpy().reshape(28, 28), cmap='binary')
            col.axis('off')
            i += 1
    if savepath is not None:
        plt.savefig(savepath)
    plt.draw()