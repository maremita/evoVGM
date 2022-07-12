import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.gamma import Gamma
from torch.distributions.log_normal import LogNormal
from torch.distributions.kl import kl_divergence


__author__ = "amine remita"


class BranchIndDeepLogNEncoder(nn.Module):

    def __init__(
            self,
            m_dim,
            h_dim,
            n_layers=3, 
            branch_prior_hp=torch.tensor([0.1, 0.1]),
            device=torch.device("cpu")):

        super().__init__()

        self.m_dim = m_dim
        self.h_dim = h_dim
        self.b_dim = 1
        self.n_layers = n_layers
        self.device_ = device

        # hyper-param for branch mean and sigma LogNormal prior
        self.prior_mu, self.prior_sigma = branch_prior_hp.to(
                self.device_)

        self.noise = torch.zeros((self.m_dim, self.b_dim)).uniform_(
                ).to(self.device_)
        #self.noise = torch.zeros((self.m_dim, self.b_dim)).normal_(
        #        ).to(self.device_)

        layers = [nn.Linear(self.b_dim, self.h_dim,
            bias=True).to(self.device_), nn.ReLU()]

        for i in range(1, self.n_layers-1):
            layers.extend([nn.Linear(self.h_dim, self.h_dim,
                bias=True).to(self.device_), nn.ReLU()])

        self.net = nn.Sequential(*layers)

        self.net_mu = nn.Linear(self.h_dim, self.b_dim).to(
                self.device_)
        #self.net_mu = nn.Sequential(
        #    nn.Linear(self.h_dim, self.b_dimp).to(self.device_),
        #    nn.LogSigmoid()) # LogSigmoid, Sigmoid, Tanh, SiLU

        self.net_sigma = nn.Sequential(
            nn.Linear(self.h_dim, self.b_dim).to(self.device_),
            nn.Softplus())

        # Prior distribution
        self.b_dist_p = LogNormal(self.prior_mu, self.prior_sigma)

    def forward(self, 
            sample_size=1,
            min_clamp=0.000001,
            max_clamp=False):

        enc = self.net(self.noise)
        b_mu = self.net_mu(enc)
        b_sigma = self.net_sigma(enc)
 
        # Approximate distribution
        b_dist_q = LogNormal(b_mu, b_sigma)

        # Sample branch lengths
        samples = b_dist_q.rsample(torch.Size([sample_size]))
        #print("samples shape {}".format(samples.shape)) 
        # [sample_size, m_dim, b_dim]
        #print(samples)

        if not isinstance(min_clamp, bool):
            if isinstance(min_clamp, (float, int)):
                samples = samples.clamp(min=min_clamp)

        if not isinstance(max_clamp, bool):
            if isinstance(max_clamp, (float, int)):
                samples = samples.clamp(max=max_clamp)
 
        b_kl = kl_divergence(b_dist_q, self.b_dist_p).view(
                self.m_dim, 1)
#         print("b_kl")
#         print(b_kl.shape) # [m_dim, 1]
#         print(b_kl)
        
        return samples, b_kl


class BranchIndGammaEncoder(nn.Module):

    def __init__(
            self,
            m_dim, 
            branch_prior_hp=torch.tensor([0.1, 0.1]),
            device=torch.device("cpu")):

        super().__init__()

        self.b_dim = 1
        self.m_dim = m_dim
        self.device_ = device

        # hyper-param for branch mean and sigma LogNormal prior
        self.prior_alpha, self.prior_rate = branch_prior_hp.to(
                self.device_)

        self.b_alpha = nn.Parameter(
                torch.zeros(self.m_dim, self.b_dim),
                requires_grad=True).to(self.device_)
        self.b_rate = nn.Parameter(
                torch.zeros(self.m_dim, self.b_dim),
                requires_grad=True).to(self.device_)

        nn.init.uniform_(self.b_alpha.data) 
        nn.init.uniform_(self.b_rate.data)

        # Prior distribution
        self.b_dist_p = Gamma(self.prior_alpha, self.prior_rate)

    def forward(self, 
            sample_size=1,
            min_clamp=0.000001,
            max_clamp=False):
        
        self.b_alpha.data = F.softplus(self.b_alpha.data)
        self.b_rate.data = F.softplus(self.b_rate.data)
        
#         print("b_alpha")
#         print(self.b_alpha.shape) # [m_dim, b_dim]
#         print(self.b_alpha)
        
        # Approximate distribution
        b_dist_q = Gamma(self.b_alpha, self.b_rate)
        
        # Sample branch lengths
        samples = b_dist_q.rsample(torch.Size([sample_size]))
        #print("samples shape {}".format(samples.shape))
        # [sample_size, m_dim, b_dim]
        #print(samples)

        if not isinstance(min_clamp, bool):
            if isinstance(min_clamp, (float, int)):
                samples = samples.clamp(min=min_clamp)

        if not isinstance(max_clamp, bool):
            if isinstance(max_clamp, (float, int)):
                samples = samples.clamp(max=max_clamp)

        b_kl = kl_divergence(b_dist_q, self.b_dist_p).view(
                self.m_dim, 1)
#         print("b_kl")
#         print(b_kl.shape) # [m_dim, 1]
#         print(b_kl)

        return samples, b_kl


class BranchIndDeepGammaEncoder(nn.Module):

    def __init__(
            self,
            m_dim,
            h_dim, 
            n_layers=3,
            branch_prior_hp=torch.tensor([0.1, 0.1]),
            device=torch.device("cpu")):

        super().__init__()

        self.m_dim = m_dim
        self.h_dim = h_dim
        self.b_dim = 1
        self.n_layers = n_layers
        self.device_ = device

        # hyper-param for branch mean and sigma LogNormal prior
        self.prior_alpha, self.prior_rate = branch_prior_hp.to(
                self.device_)

        self.noise = torch.zeros((self.m_dim, self.b_dim)).uniform_(
                ).to(self.device_)
        #self.noise = torch.zeros((self.m_dim, self.b_dim)).normal_(
        #        ).to(self.device_)
        #print("self.noise", self.noise.device)

        layers = [nn.Linear(self.b_dim, self.h_dim,
            bias=True).to(self.device_), nn.ReLU()]

        for i in range(1, self.n_layers-1):
            layers.extend([nn.Linear(self.h_dim, self.h_dim,
                bias=True).to(self.device_), nn.ReLU()])

        self.net = nn.Sequential(*layers)

        self.net_alpha = nn.Sequential(
            nn.Linear(self.h_dim, self.b_dim).to(self.device_),
            nn.Softplus()) # Sigmoid,

        self.net_rate = nn.Sequential(
            nn.Linear(self.h_dim, self.b_dim).to(self.device_),
            nn.Softplus()) 

        # Prior distribution
        self.b_dist_p = Gamma(self.prior_alpha, self.prior_rate)

    def forward(self, 
            sample_size=1,
            min_clamp=0.000001,
            max_clamp=False):

        enc = self.net(self.noise)
        #print("self.noise", self.noise.device)
        #print("enc", enc.device)
        b_alpha = self.net_alpha(enc) #* 10 #.clamp(max=10.)
        b_rate = self.net_rate(enc) #.pow(2) #.clamp(max=100.)

#         print("b_alpha")
#         print(b_alpha.shape) # [m_dim, b_dim]
        #print("b_alpha ", b_alpha.device)

        # Approximate distribution
        b_dist_q = Gamma(b_alpha, b_rate)

        samples = b_dist_q.rsample(torch.Size([sample_size]))
        #print("samples shape {}".format(samples.shape))
        ## [sample_size, m_dim, b_dim]
        #print("samples", samples.device)

        if not isinstance(min_clamp, bool):
            if isinstance(min_clamp, (float, int)):
                samples = samples.clamp(min=min_clamp)

        if not isinstance(max_clamp, bool):
            if isinstance(max_clamp, (float, int)):
                samples = samples.clamp(max=max_clamp)

        b_kl = kl_divergence(b_dist_q, self.b_dist_p).view(
                self.m_dim, 1)
#         print("b_kl")
#         print(b_kl.shape) # [m_dim, 1]
#         print(b_kl)

        return samples, b_kl
