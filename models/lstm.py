import torch
from torch import nn


class LSTM(nn.Module):
    
    def __init__(self, in_channels, feature_dim, hidden_dim=64, layer_num=2):
        super(LSTM, self).__init__()
        
        # (batchsize,1,channel_num,feature_dim)-->(batchsize,1,channel_num*feature_dim)
        self.flatten = nn.Flatten(start_dim=2)
        
        # (batchsize,1,channel_num*feature_dim)-->(batchsize, hidden_dim)
        self.layer_num = layer_num
        self.hidden_dim = hidden_dim
        self.cell = nn.LSTM(
            input_size=in_channels*feature_dim, 
            hidden_size=hidden_dim, 
            num_layers=layer_num, 
            dropout=0.01, 
            batch_first=True # input: (batch_size, squences, features)
            )
        
        # (batchsize, hidden_dim)-->(batchsize, channel_num*feature_dim)
        self.fc = nn.Linear(hidden_dim, in_channels*feature_dim)
        
        # (batchsize, channel_num*feature_dim)-->(batchsize,1,channel_num,feature_dim)
        self.unflatten = nn.Unflatten(-1, (1, in_channels, feature_dim))

        # scale inside the model
        self.register_buffer('min', torch.zeros(in_channels, feature_dim, dtype=torch.float32))
        self.register_buffer('max', torch.ones(in_channels, feature_dim, dtype=torch.float32))
    
    def forward(self, x, device=torch.device('cuda:1')):
        
        h0 = torch.zeros(self.layer_num * 1, len(x), self.hidden_dim, dtype=torch.float32, device=device)
        c0 = torch.zeros(self.layer_num * 1, len(x), self.hidden_dim, dtype=torch.float32, device=device)
        
        x = self.flatten(x)
        _, (h, c)  = self.cell(x, (h0, c0))
        y = self.fc(h[-1])
        y = self.unflatten(y)
        
        return y
    
    def scale(self, x):
        return (x-self.min) / (self.max-self.min+1e-6)
    
    def descale(self, x):
        return x * (self.max-self.min+1e-6) + self.min