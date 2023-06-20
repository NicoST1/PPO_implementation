import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Network(nn.Module):
    def __init__(self, dims, activation='relu') -> None:
        super(Network, self).__init__()

        # check that activation function is valid
        assert(activation in ['relu', 'tanh'])
        self.activation = torch.tanh if activation == 'tanh' else F.relu

        self.layers = len(dims)
        self.l1 = nn.Linear(dims[0], dims[1])
        self.l2 = nn.Linear(dims[1], dims[2])
        self.l3 = nn.Linear(dims[2], dims[3])

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float)

        a1 = self.activation(self.l1(state))
        a2 = self.activation(self.l2(a1))
        out = self.l3(a2)
        return out
