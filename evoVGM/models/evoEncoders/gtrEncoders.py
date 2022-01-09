import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.dirichlet import Dirichlet
from torch.distributions.kl import kl_divergence

__author__ = "a.r."


class GTRSubRateIndDirEncoder(nn.Module):

    def __init__(
            self,
            m_dim,
            rates_prior=torch.ones(6),
            device=torch.device("cpu")):

        super().__init__()

        self.m_dim = m_dim
        self.r_dim = 6      # AG, AC, AT, GC, GT, CT 
        self.device = device

        self.pr = rates_prior # Hyper-param for rate prior

        self.rates = nn.Parameter(torch.zeros(self.r_dim), requires_grad=True)
        nn.init.uniform_(self.rates.data) # uniform_ xavier_uniform_
#         nn.init.xavier_uniform_(self.rates.data) # uniform_ xavier_uniform_


    def forward(self, sample_size):
        
        self.rates.data = F.softplus(self.rates.data)
#         print("rates")
#         print(self.rates.shape) # [6]
#         print(self.rates)

        # Prior distribution
        r_dist_p = Dirichlet(self.pr)
        
        # Approximate distribution
        r_dist_q = Dirichlet(self.rates)

        r_samples = r_dist_q.rsample(torch.Size([sample_size]))
#         print("r_samples shape {}".format(r_samples.shape)) # [sample_size, r_dim]
#         print(r_samples)

        r_kl = kl_divergence(r_dist_q, r_dist_p).expand(self.m_dim, 1) #.flatten()
#         print("r_kl")
#         print(r_kl.shape) # [m_dim, 1]
#         print(r_kl)

        return r_samples, r_kl


class GTRSubRateIndDeepDirEncoder(nn.Module):

    def __init__(
            self,
            m_dim,
            h_dim, 
            n_layers=2,
            rates_prior=torch.ones(6),
            device=torch.device("cpu")):

        super().__init__()

        self.m_dim = m_dim
        self.h_dim = h_dim
        self.in_dim = 6     # AG, AC, AT, GC, GT, CT
        self.r_dim = 6     # AG, AC, AT, GC, GT, CT
        self.n_layers = n_layers
        self.device = device

        self.pr = rates_prior # Hyper-param for rate prior

        self.noise = torch.ones((self.in_dim)).uniform_()
#         self.noise = torch.ones((self.r_dim)).normal_()

        layers = [nn.Linear(self.in_dim, self.h_dim, bias=True), nn.ReLU()]

        for i in range(1, self.n_layers-1):
            layers.extend([nn.Linear(self.h_dim, self.h_dim, bias=True), nn.ReLU()])

        layers.extend([nn.Linear(self.h_dim, self.r_dim, bias=True), nn.Softplus()])

        self.net = nn.Sequential(*layers)

    def forward(self, sample_size):
        
        rates = self.net(self.noise)
#         print("rates")
#         print(rates.shape) # [6]
#         print(rates)

        # Prior distribution
        r_dist_p = Dirichlet(self.pr)
        
        # Approximate distribution
        r_dist_q = Dirichlet(rates)
        
#         r_samples = r_dist_q.rsample(torch.Size([sample_size]))
        r_samples = r_dist_q.rsample(torch.Size([sample_size]))
#         print("r_samples shape {}".format(r_samples.shape)) # [sample_size, r_dim]
#         print(r_samples)

        r_kl = kl_divergence(r_dist_q, r_dist_p).expand(self.m_dim, 1) #.flatten()
#         print("r_kl")
#         print(r_kl.shape) # [m_dim, 1]
#         print(r_kl)

        return r_samples, r_kl


class GTRfreqIndDeepDirEncoder(nn.Module):

    def __init__(
            self,
            m_dim,
            h_dim, 
            n_layers=2, 
            freqs_prior=torch.ones(4),
            device=torch.device("cpu")):

        super().__init__()

        self.m_dim = m_dim
        self.h_dim = h_dim
        self.in_dim = 4
        self.f_dim = 4
        self.n_layers = n_layers
        self.device = device

        self.pi = freqs_prior # Hyper-param for rate prior

        self.noise = torch.ones((self.in_dim)).uniform_()
#         self.noise = torch.ones((self.f_dim)).normal_()

        layers = [nn.Linear(self.in_dim, self.h_dim, bias=True), nn.ReLU()]

        for i in range(1, self.n_layers-1):
            layers.extend([nn.Linear(self.h_dim, self.h_dim, bias=True), nn.ReLU()])

        layers.extend([nn.Linear(self.h_dim, self.f_dim, bias=True), nn.Softplus()])

        self.net = nn.Sequential(*layers)

    def forward(self, sample_size):
        
        freqs = self.net(self.noise)
#         print("freqs")
#         print(freqs.shape) # [f_dim]
#         print(freqs)

        # Prior distribution
        f_dist_p = Dirichlet(self.pi)
        
        # Approximate distribution
        f_dist_q = Dirichlet(freqs)
        
        f_samples = f_dist_q.rsample(torch.Size([sample_size]))
#         print("f_samples shape {}".format(f_samples.shape)) # [sample_size, f_dim]
#         print(f_samples)

        f_kl = kl_divergence(f_dist_q, f_dist_p).expand(self.m_dim, 1) #.flatten()
#         print("f_kl")
#         print(f_kl.shape) # [m_dim, 1]
#         print(f_kl)

        return f_samples, f_kl

