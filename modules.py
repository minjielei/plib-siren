import torch
from torch import nn

import numpy as np

class Sine(nn.Module):
    def __init__(self, in_features, out_features, first_layer, bias=True, omega=30):
        super().__init__()
        self.omega = omega
        self.first_layer = first_layer
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.first_layer:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                # See supplement Sec. 1.5 for discussion of factor 30
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega, np.sqrt(6 / self.in_features) / self.omega)

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(self.omega * self.linear(input))

class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, omega=30):
        super().__init__()

        self.net = []
        self.net.append(Sine(in_features, hidden_features, first_layer=True, omega=omega))

        for i in range(hidden_layers):
            self.net.append(Sine(hidden_features, hidden_features, first_layer=False, omega=omega))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / omega,
                                             np.sqrt(6 / hidden_features) / omega)

            self.net.append(final_linear)
        else:
            self.net.append(Sine(hidden_features, out_features, first_layer=False, omega=omega))

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        coords = input.clone().detach().requires_grad_(True)  # allows to take derivative w.r.t. input
        output = self.net(coords.float())
        return {'model_out' : output, 'model_in' : coords}



