import torch
from torch import nn


class TimeLaggedAE(nn.Module):
    
    def __init__(self, in_channels, feature_dim, embed_dim):
        super(TimeLaggedAE, self).__init__()
                
        # (batchsize,1,channel_num,feature_dim)-->(batchsize, embed_dim)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels*feature_dim, 64, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(64, embed_dim, bias=True),
        )
        
        # (batchsize, embed_dim)-->(batchsize,1,channel_num,feature_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 64, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(64, in_channels*feature_dim, bias=True),
            nn.Unflatten(-1, (1, in_channels, feature_dim))
        )
        
        # scale inside the model
        self.register_buffer('min', torch.zeros(in_channels, feature_dim, dtype=torch.float32))
        self.register_buffer('max', torch.ones(in_channels, feature_dim, dtype=torch.float32))
        
    def forward(self,x):
        embed = self.encoder(x)
        out = self.decoder(embed)
        return out, embed

    def scale(self, x):
        return (x-self.min) / (self.max-self.min+1e-6)
    
    def descale(self, x):
        return x * (self.max-self.min+1e-6) + self.min