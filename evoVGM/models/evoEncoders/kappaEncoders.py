import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.gamma import Gamma
#from torch.distributions.log_normal import LogNormal
from torch.distributions.kl import kl_divergence


__author__ = "amine remita"


class KappaIndDeepGammaEncoder(nn.Module):

    def __init__(
            self,
            h_dim, 
            n_layers=3,
            k_prior=torch.tensor([0.1]),
            device=torch.device("cpu")):

        super().__init__()

        self.h_dim = h_dim
        self.k_dim = 1
        self.n_layers = n_layers
        self.device = device

        # hyper-param for branch mean and sigma LogNormal prior
        self.prior_alpha, self.prior_rate = k_prior
        
        self.noise = torch.ones((self.k_dim)).uniform_()
#         self.noise = torch.ones((self.k_dim)).normal_()

        layers = [nn.Linear(self.k_dim, self.h_dim, bias=True),
                nn.ReLU()]

        for i in range(1, self.n_layers-1):
            layers.extend([nn.Linear(self.h_dim, self.h_dim,
                bias=True), nn.ReLU()])

        self.net = nn.Sequential(*layers)

        self.net_alpha = nn.Sequential(
            nn.Linear(self.h_dim, self.k_dim),
            nn.Softplus()) # Sigmoid,

        self.net_rate = nn.Sequential(
            nn.Linear(self.h_dim, self.k_dim),
            nn.Softplus()) 

        # Prior distribution
        self.dist_p = Gamma(self.prior_alpha, self.prior_rate)

    def forward(self, 
            sample_size=1,
            min_clamp=0.000001,
            max_clamp=False):

        enc = self.net(self.noise)
        alpha = self.net_alpha(enc) #* 10 #.clamp(max=10.)
        rate = self.net_rate(enc) #.pow(2) #.clamp(max=100.)

#         print("alpha")
#         print(alpha.shape) # [k_dim]
#         print(alpha)

        # Approximate distribution
        dist_q = Gamma(alpha, rate)

        samples = dist_q.rsample(torch.Size([sample_size]))
#         print("samples shape {}".format(samples.shape))
        # [sample_size, k_dim]
#         print(samples)

        if not isinstance(min_clamp, bool):
            if isinstance(min_clamp, (float, int)):
                samples = samples.clamp(min=min_clamp)

        if not isinstance(max_clamp, bool):
            if isinstance(max_clamp, (float, int)):
                samples = samples.clamp(max=max_clamp)

        kl = kl_divergence(dist_q, self.dist_p).flatten()
#         print("kl")
#         print(kl.shape) # [1]
#         print(kl)

        return samples, kl
