from network import Network
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical


class Actor(Network):
    def __init__(self, dims, activation='tanh') -> None:
        super().__init__(dims, activation)

    def forward(self, state, softmax_dim=0):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float)

        a1 = self.activation(self.l1(state))
        a2 = self.activation(self.l2(a1))
        a3 = self.l3(a2)
        out = F.softmax(a3, dim=softmax_dim)
        return out

    def sample_action(self, state, softmax_dim=0):
        with torch.no_grad():
            action_probs = self.forward(state, softmax_dim)
            m = Categorical(action_probs)
            action = m.sample().item()
            action_p = action_probs[action].item()
        return action, action_p

    def get_best_action(self, state):
        with torch.no_grad():
            action_probs = self.forward(state)
            action = torch.argmax(action_probs).item()
        return action, 1.0


class Critic(Network):
    def __init__(self, dims, activation='relu') -> None:
        super().__init__(dims, activation)

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float)

        a1 = self.activation(self.l1(state))
        a2 = self.activation(self.l2(a1))
        out = self.l3(a2)
        return out

    def evaluate(self, states):
        with torch.no_grad():
            return self.forward(states)


