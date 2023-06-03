from torch import nn
from .tcn import TCN
from .lstm import LSTM
from .neural_ode import NeuralODE
from .time_lagged import TimeLaggedAE
from .slow_fast_evolve import DynamicsEvolver


def weights_normal_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None: nn.init.zeros_(m.bias)