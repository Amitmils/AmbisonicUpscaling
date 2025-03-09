import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class HPNet(nn.Module):
    def __init__(
        self,
        mode: str,
        sources: int,
        num_iters: int,
        freq_bins: int,
        T: int,
        input_ambi_channels: int,
        mu : float,
        ro : float,
        version : int = 0
    ):
        super().__init__()
        self.mode = mode
        self.device = 'cuda'
        if self.mode == 'classic':
            self.mu = torch.tensor(mu).log()
            self.ro = torch.tensor(ro).log()

        elif self.mode == "DU_simple":
            #[Time Group, Bin, SOURCES, T*2] ---> to add Batch aswell
            if version == 0:
                self.mu = nn.Parameter(torch.full((1,1,1,1),math.log(mu)),requires_grad=True) #to keep them positive, we use exp(mu)
                self.ro = nn.Parameter(torch.full((1,1,1,1),math.log(ro)),requires_grad=True) #to keep them positive, we use exp(ro)
            elif version == 1:
                self.mu = nn.Parameter(torch.full((1,freq_bins,1,1),math.log(mu)),requires_grad=True) #to keep them positive, we use exp(mu)
                self.ro = nn.Parameter(torch.full((1,freq_bins,1,1),math.log(ro)),requires_grad=True) #to keep them positive, we use exp(ro)
            elif version == 2:
                self.mu = nn.Parameter(torch.full((1,freq_bins,1,T*2),math.log(mu)),requires_grad=True) #to keep them positive, we use exp(mu)
                self.ro = nn.Parameter(torch.full((1,freq_bins,1,T*2),math.log(ro)),requires_grad=True) #to keep them positive, we use exp(ro)
            elif version == 3:
                self.mu = nn.Parameter(torch.full((1,freq_bins,1,T*2),math.log(mu)),requires_grad=True) #to keep them positive, we use exp(mu)
                self.ro = nn.Parameter(torch.full((1,freq_bins,input_ambi_channels,T*2),math.log(ro)),requires_grad=True) #to keep them positive, we use exp(ro)
            elif version == 4:
                self.mu = nn.Parameter(torch.full((1,freq_bins,sources,T*2),math.log(mu)),requires_grad=True) #to keep them positive, we use exp(mu)
                self.ro = nn.Parameter(torch.full((1,freq_bins,input_ambi_channels,T*2),math.log(ro)),requires_grad=True) #to keep them positive, we use exp(ro)
            if version == 5:
                self.mu = nn.Parameter(torch.full((num_iters,1,1,1,1),math.log(mu)),requires_grad=True) #to keep them positive, we use exp(mu)
                self.ro = nn.Parameter(torch.full((num_iters,1,1,1,1),math.log(ro)),requires_grad=True) #to keep them positive, we use exp(ro)

        elif self.mode == "DU_attention":
            pass


    def forward(self,iter):
        if self.mode == 'classic':
            if self.mu.shape[0] == 1:
                return  self.mu,self.ro
            else:
                return  self.mu[iter],self.ro[iter]

        if self.mode == 'DU_simple':
            
            if self.mu.dim() != 5: #wont hold if we add batch dim
                return  self.mu,self.ro
            else:
                return  self.mu[iter],self.ro[iter]
