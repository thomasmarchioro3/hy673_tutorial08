import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam

from torch.distributions import Normal

from ex1_0_utils import LogisticDistribution, generate_from_model


"""
Thrupling layer (Coupling layer with triple split)
"""
class ThruplingLayer(nn.Module):

    def __init__(self, input_dim, output_dim=None, hidden_dim=256, masks=None, num_layers=4):
        super().__init__()

        self.input_dim = input_dim
        if output_dim is None:
            self.output_dim = input_dim
        else:
            self.output_dim = output_dim

        self.hidden_dim = hidden_dim

        if masks is None:
            raise ValueError("Error: Mask cannot be empty")

        self.masks = masks

        modules_m = [nn.Linear(input_dim, hidden_dim), 
                    nn.LeakyReLU(0.2)]

        for _ in range(num_layers - 2):
            modules_m.append(nn.Linear(hidden_dim, hidden_dim))
            modules_m.append(nn.LeakyReLU(0.2))
        modules_m.append(nn.Linear(hidden_dim, self.output_dim))

        self.m = nn.Sequential(*modules_m)

        modules_l = [nn.Linear(input_dim, hidden_dim), 
                    nn.LeakyReLU(0.2)]

        for _ in range(num_layers - 2):
            modules_l.append(nn.Linear(hidden_dim, hidden_dim))
            modules_l.append(nn.LeakyReLU(0.2))
        modules_l.append(nn.Linear(hidden_dim, self.output_dim))

        self.l = nn.Sequential(*modules_l)

    def forward(self, z):
        z_A = self.masks[0] * z
        z_B = self.masks[1] * z
        z_C = self.masks[2] * z
        x_A = z_A
        
        x_B = z_B + self.m(z_A)*self.masks[1]
        x_C = z_C + self.l(z_A + z_B)*self.masks[2]
        return x_A + x_B + x_C
    
    # inverse mapping
    def inverse(self, x):
        x_A = self.masks[0] * x
        x_B = self.masks[1] * x
        x_C = self.masks[2] * x
        z_A = x_A
        z_B = x_B - self.m(z_A)*self.masks[1]
        z_C = x_C - self.l(z_A + z_B)*self.masks[2]
        return z_A + z_B + z_C
    
"""
Scaling layer with constant term to be applied at the end of the network
"""
class ScalingLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.log_scale_vector = nn.Parameter(torch.randn(1, input_dim, requires_grad=True))


    def forward(self, x):
        return torch.exp(self.log_scale_vector) * x

    def inverse(self, z):
        log_det_jacobian = torch.sum(-self.log_scale_vector, dim=-1)
        x = torch.exp(-self.log_scale_vector) * z
        return x, log_det_jacobian
  


"""
NICER: NICE with triple-split coupling layers
"""
class NICER(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_coupling_layers=3, num_layers=6, device='cpu', 
                 prior_type='normal', use_scaling=True):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_coupling_layers = num_coupling_layers
        self.num_layers = num_layers  # number of linear layers for each coupling layer
        self.prior_type = prior_type
        self.use_scaling = use_scaling

        # alternating mask orientations for consecutive coupling layers
        masks = [self._get_masks(input_dim, shift=(i % 3 == 0)).to(device)
                                                for i in range(num_coupling_layers)]

        self.coupling_layers = nn.ModuleList([ThruplingLayer(input_dim=input_dim,
                                    hidden_dim=hidden_dim,
                                    masks=masks[i], num_layers=num_layers)
                                for i in range(num_coupling_layers)])
        if use_scaling:
            self.scaling_layer = ScalingLayer(input_dim=input_dim)

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
        if self.use_scaling:
            x = self.scaling_layer(x)
        return x

    def inverse(self, x):
        z = x
        log_likelihood = 0
        if self.use_scaling:
            z, log_det_jacobian = self.scaling_layer.inverse(z)
            log_likelihood += log_det_jacobian
        for i in reversed(range(len(self.coupling_layers))):  # pass through each coupling layer in reversed order
            z = self.coupling_layers[i].inverse(z)
        
        log_likelihood = torch.sum(self.prior.log_prob(z), dim=-1) + log_likelihood

        return z, log_likelihood

    def sample(self, num_samples):
        z = self.prior.sample([num_samples, self.input_dim]).view(num_samples, self.input_dim)
        z = z.to(self.device)
        return self.forward(z)

    def _get_masks(self, dim, shift=0):
        shift = shift % 3
        mask = torch.zeros((3, dim))
        for i in range(3):
            curr_shift = (shift + i) % 3
            mask[i, curr_shift::3] = 1.
        return mask.float()
