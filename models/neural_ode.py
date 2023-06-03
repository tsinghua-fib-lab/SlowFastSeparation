import torch
from torch import nn
if True:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


class NeuralODEfunc(nn.Module):

    def __init__(self, obs_dim, nhidden=64):
        super(NeuralODEfunc, self).__init__()
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(obs_dim, nhidden)
        self.fc2 = nn.Linear(nhidden, obs_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        return out


class NeuralODE(nn.Module):

    def __init__(self, in_channels, feature_dim, nhidden=64):
        super(NeuralODE, self).__init__()

        self.ode = NeuralODEfunc(in_channels*feature_dim, nhidden)

        self.flatten = nn.Flatten(start_dim=-2)
        self.unflatten = nn.Unflatten(-1, (in_channels, feature_dim))

        # scale inside the model
        self.register_buffer('min', torch.zeros(in_channels, feature_dim, dtype=torch.float32))
        self.register_buffer('max', torch.ones(in_channels, feature_dim, dtype=torch.float32))
    
    def forward(self, x0, t):
        x0 = self.flatten(x0)[:,0]
        out = odeint(self.ode, x0, t).permute(1, 0, 2)
        out = self.unflatten(out)
        return out

    def scale(self, x):
        return (x-self.min) / (self.max-self.min+1e-6)
    
    def descale(self, x):
        return x * (self.max-self.min+1e-6) + self.min