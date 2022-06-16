import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.dirichlet import Dirichlet
from torch.distributions.kl import kl_divergence

__author__ = "amine remita"


class GTRSubRateIndDirEncoder(nn.Module):

    def __init__(
            self,
            m_dim,
            rates_prior=torch.ones(6),
            device=torch.device("cpu")):

        super().__init__()

        self.m_dim = m_dim
        self.r_dim = 6      # AG, AC, AT, GC, GT, CT 
        self.device_ = device

        # Hyper-param for rate prior
        self.pr = rates_prior.to(self.device_)

        self.rates = nn.Parameter(
                torch.zeros(self.r_dim),
                requires_grad=True).to(self.device_)
        nn.init.uniform_(self.rates.data) # uniform_ xavier_uniform_

        # Prior distribution
        self.r_dist_p = Dirichlet(self.pr)
        
    def forward(
            self, 
            sample_size=1,
            min_clamp=False,    # should be <= to 10^-7
            max_clamp=False):
 
        self.rates.data = F.softplus(self.rates.data)
#         print("rates")
#         print(self.rates.shape) # [6]
#         print(self.rates)

        # Approximate distribution
        r_dist_q = Dirichlet(self.rates)

        samples = r_dist_q.rsample(torch.Size([sample_size]))
#         print("samples shape {}".format(samples.shape)) # [sample_size, r_dim]
#         print(samples)

        if not isinstance(min_clamp, bool):
            if isinstance(min_clamp, (float, int)):
                samples = samples.clamp(min=min_clamp)

        if not isinstance(max_clamp, bool):
            if isinstance(max_clamp, (float, int)):
                samples = samples.clamp(max=max_clamp)

        r_kl = kl_divergence(r_dist_q, self.r_dist_p).flatten()
#         print("r_kl")
#         print(r_kl.shape) # [m_dim, 1]
#         print(r_kl)

        return samples, r_kl


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
        self.device_ = device

        # Hyper-param for rate prior
        self.pr = rates_prior.to(self.device_)

        self.noise = torch.zeros((self.in_dim)).uniform_(
                ).to(self.device_)
        #self.noise = torch.zeros((self.in_dim)).normal_(
        #        ).to(self.device_)

        layers = [nn.Linear(self.in_dim, self.h_dim,
            bias=True).to(self.device_), nn.ReLU()]

        for i in range(1, self.n_layers-1):
            layers.extend([nn.Linear(self.h_dim, self.h_dim,
                bias=True).to(self.device_), nn.ReLU()])

        layers.extend([nn.Linear(self.h_dim, self.r_dim,
            bias=True).to(self.device_), nn.Softplus()])

        self.net = nn.Sequential(*layers)

        # Prior distribution
        self.r_dist_p = Dirichlet(self.pr)
        
    def forward(
            self, 
            sample_size=1,
            min_clamp=False,    # should be <= to 10^-7
            max_clamp=False):
        
        rates = self.net(self.noise)
#         print("rates")
#         print(rates.shape) # [6]
#         print(rates)

        # Approximate distribution
        r_dist_q = Dirichlet(rates)
        
#         samples = r_dist_q.rsample(torch.Size([sample_size]))
        samples = r_dist_q.rsample(torch.Size([sample_size]))
#         print("samples shape {}".format(samples.shape)) # [sample_size, r_dim]
#         print(samples)

        if not isinstance(min_clamp, bool):
            if isinstance(min_clamp, (float, int)):
                samples = samples.clamp(min=min_clamp)

        if not isinstance(max_clamp, bool):
            if isinstance(max_clamp, (float, int)):
                samples = samples.clamp(max=max_clamp)

        r_kl = kl_divergence(r_dist_q, self.r_dist_p).flatten()
#         print("r_kl")
#         print(r_kl.shape) # [m_dim, 1]
#         print(r_kl)

        return samples, r_kl


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
        self.device_ = device

        # Hyper-param for rate prior
        self.pi = freqs_prior.to(self.device_) 

        self.noise = torch.zeros((self.in_dim)).uniform_(
                ).to(self.device_)
        #self.noise = torch.zeros((self.in_dim)).normal_(
        #        ).to(self.device_)

        layers = [nn.Linear(self.in_dim, self.h_dim,
            bias=True).to(self.device_), nn.ReLU()]

        for i in range(1, self.n_layers-1):
            layers.extend([nn.Linear(self.h_dim, self.h_dim,
                bias=True).to(self.device_), nn.ReLU()])

        layers.extend([nn.Linear(self.h_dim, self.f_dim,
            bias=True).to(self.device_), nn.Softplus()])

        self.net = nn.Sequential(*layers)

        # Prior distribution
        self.f_dist_p = Dirichlet(self.pi)
 
    def forward(
            self, 
            sample_size=1,
            min_clamp=False,    # should be <= to 10^-7
            max_clamp=False):
        
        freqs = self.net(self.noise)
#         print("freqs")
#         print(freqs.shape) # [f_dim]
#         print(freqs)

        # Approximate distribution
        f_dist_q = Dirichlet(freqs)
        
        samples = f_dist_q.rsample(torch.Size([sample_size]))
        #print("samples shape {}".format(samples.shape))
        # [sample_size, f_dim]
        #print(samples)

        if not isinstance(min_clamp, bool):
            if isinstance(min_clamp, (float, int)):
                samples = samples.clamp(min=min_clamp)

        if not isinstance(max_clamp, bool):
            if isinstance(max_clamp, (float, int)):
                samples = samples.clamp(max=max_clamp)

        f_kl = kl_divergence(f_dist_q, self.f_dist_p).flatten()
        #print("f_kl")
        #print(f_kl.shape) # [1]
        #print(f_kl)

        return samples, f_kl

