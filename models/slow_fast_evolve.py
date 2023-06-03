import torch
from torch import nn


class Koopman_OPT(nn.Module):

    def __init__(self, koopman_dim):
        super(Koopman_OPT, self).__init__()

        self.koopman_dim = koopman_dim

        self.register_parameter('V', nn.Parameter(torch.Tensor(koopman_dim, koopman_dim)))
        self.register_parameter('Lambda', nn.Parameter(torch.Tensor(koopman_dim)))

        # init
        torch.nn.init.normal_(self.V, mean=0, std=0.1)
        torch.nn.init.normal_(self.Lambda, mean=0, std=0.1)

    def forward(self, tau):

        tmp1 = torch.diag(torch.exp(tau*self.Lambda))
        V_inverse = torch.inverse(self.V)
        K = torch.mm(torch.mm(V_inverse, tmp1), self.V)

        return K
        # return tmp1
    

class LSTM_OPT(nn.Module):
    
    def __init__(self, in_channels, feature_dim, hidden_dim, layer_num, tau_s, device):
        super(LSTM_OPT, self).__init__()
        
        self.device = device
        self.tau_s = tau_s
        
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
            batch_first=True  # input: (batch_size, squences, features)
            )

        self.eta_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # constraint to be greater than 0
        )
        self.ksi_head = nn.Linear(hidden_dim, in_channels*feature_dim)
        
        # (batchsize, channel_num*feature_dim)-->(batchsize,1,channel_num,feature_dim)
        self.unflatten = nn.Unflatten(-1, (in_channels, feature_dim))
    
    def forward(self, x):
        
        h0 = torch.zeros(self.layer_num * 1, len(x), self.hidden_dim, dtype=torch.float32, device=self.device)
        c0 = torch.zeros(self.layer_num * 1, len(x), self.hidden_dim, dtype=torch.float32, device=self.device)
        
        x = self.flatten(x)
        _, (h, c)  = self.cell(x, (h0, c0))

        eta = self.eta_head(h[-1]).unsqueeze(-2)  # (batch_size, 1, 1)
        ksi = self.ksi_head(h[-1]).unsqueeze(-2)  # (batch_size, 1, in_channels*feature_dim)
        y = torch.abs(x) * torch.exp(-(eta+self.tau_s)) * ksi  # (batch_size, 1, in_channels*feature_dim)

        y = self.unflatten(y)
                
        return y


class DynamicsEvolver(nn.Module):
    
    def __init__(self, in_channels, feature_dim, embed_dim, slow_dim, redundant_dim, tau_s, fast, device):
        super(DynamicsEvolver, self).__init__()

        self.slow_dim = slow_dim
        
        # (batchsize,1,channel_num,feature_dim)-->(batchsize, embed_dim)
        self.encoder_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels*feature_dim, 64, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(64, embed_dim, bias=True),
        )
        
        # (batchsize, embed_dim)-->(batchsize, slow_dim)
        self.encoder_2 = nn.Sequential(
            nn.Linear(embed_dim, 64, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(64, slow_dim, bias=True),
        )

        # (batchsize, slow_dim)-->(batchsize, redundant_dim)
        self.encoder_3 = nn.Sequential(
            nn.Linear(slow_dim, 32, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(32, redundant_dim, bias=True),
        )
        
        # (batchsize, slow_dim)-->(batchsize,1,channel_num,feature_dim)
        self.decoder = nn.Sequential(
            nn.Linear(slow_dim, 64, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(64, embed_dim, bias=True),
            nn.Tanh(),
            nn.Linear(embed_dim, 64, bias=True),
            nn.Tanh(),
            nn.Dropout(p=0.01),
            nn.Linear(64, in_channels*feature_dim, bias=True),
            nn.Unflatten(-1, (1, in_channels, feature_dim))
        )
        
        self.K_opt = Koopman_OPT(slow_dim+redundant_dim)

        self.fast = fast
        self.lstm = LSTM_OPT(in_channels, feature_dim, hidden_dim=64, layer_num=2, tau_s=tau_s, device=device)

        # scale inside the model
        self.register_buffer('min', torch.zeros(in_channels, feature_dim, dtype=torch.float32))
        self.register_buffer('max', torch.ones(in_channels, feature_dim, dtype=torch.float32))
    
    def obs2slow(self, obs):
        # (batchsize,1,channel_num,feature_dim)-->(batchsize, embed_dim)-->(batchsize, slow_dim)
        embed = self.encoder_1(obs)
        slow_var = self.encoder_2(embed)
        return slow_var, embed

    def slow2obs(self, slow_var):
        # (batchsize, slow_dim)-->(batchsize,1,channel_num,feature_dim)
        obs = self.decoder(slow_var)
        return obs

    def slow2koopman(self, slow_var):
        # (batchsize, slow_dim)-->(batchsize, koopman_dim = slow_dim + redundant_dim)
        redundant_var = self.encoder_3(slow_var)
        koopman_var = torch.concat((slow_var, redundant_var), dim=-1)
        return koopman_var

    def koopman2slow(self, koopman_var):
        # (batchsize, koopman_dim = slow_dim + redundant_dim)-->(batchsize, slow_dim)
        slow_var = koopman_var[:,:self.slow_dim]
        return slow_var

    def koopman_evolve(self, koopman_var, tau=1.):
        
        K = self.K_opt(tau)
        koopman_var_next = torch.matmul(K, koopman_var.unsqueeze(-1)).squeeze(-1)
        
        return koopman_var_next
    
    def lstm_evolve(self, x, T=1):
        
        if self.fast:
            y = [self.lstm(x)]
            for _ in range(1, T-1): 
                y.append(self.lstm(y[-1]))
        else:
            y = torch.zeros_like(x)  # not predict fast dynamics

        return y[-1], y[:-1]
    
    def scale(self, x):
        return (x-self.min) / (self.max-self.min+1e-6)
    
    def descale(self, x):
        return x * (self.max-self.min+1e-6) + self.min