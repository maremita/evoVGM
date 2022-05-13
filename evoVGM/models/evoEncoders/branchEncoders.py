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
            b_prior=torch.tensor([0.1, 0.1]),
            device=torch.device("cpu")):

        super().__init__()

        self.m_dim = m_dim
        self.h_dim = h_dim
        self.b_dim = 1
        self.n_layers = n_layers
        self.device = device

        # hyper-param for branch mean and sigma LogNormal prior
        self.prior_mu, self.prior_sigma = b_prior

        self.noise = torch.zeros((self.m_dim, self.b_dim)).uniform_()
#         self.noise = torch.zeros((self.m_dim, self.b_dim)).normal_()

        layers = [nn.Linear(self.b_dim, self.h_dim, bias=True), nn.ReLU()]

        for i in range(1, self.n_layers-1):
            layers.extend([nn.Linear(self.h_dim, self.h_dim, bias=True), nn.ReLU()])

        self.net = nn.Sequential(*layers)

#         self.net_mu = nn.Linear(self.h_dim, self.b_dim)
        self.net_mu = nn.Sequential(
            nn.Linear(self.h_dim, self.b_dim),
            nn.LogSigmoid()) # LogSigmoid, Sigmoid, Tanh, SiLU

#         self.net_sigma = nn.Linear(self.h_dim, self.b_dim)
        self.net_sigma = nn.Sequential(
            nn.Linear(self.h_dim, self.b_dim),
            nn.Softplus())

    def forward(self, sample_size):

        enc = self.net(self.noise)
        b_mu = self.net_mu(enc)
        b_sigma = self.net_sigma(enc)
        
        # Prior distribution
        b_dist_p = LogNormal(self.prior_mu, self.prior_sigma)

        # Approximate distribution
        b_dist_q = LogNormal(b_mu, b_sigma)

        # Sample branch lengths
        #b_samples = self.sample(b_mu, b_sigma, sample_size)
        b_samples = b_dist_q.rsample(torch.Size([sample_size]))
#         print("b_samples shape {}".format(b_samples.shape)) # [sample_size, m_dim, b_dim]
#         print(b_samples)

        b_kl = kl_divergence(b_dist_q, b_dist_p).view(self.m_dim, 1)
#         print("b_kl")
#         print(b_kl.shape) # [m_dim, 1]
#         print(b_kl)
        
        return b_samples, b_kl

    #def sample(self, mean, std, n_sample):
    #    # Reparameterized sampling of lognormal
    #    eps = torch.FloatTensor(torch.Size([n_sample, self.m_dim, self.b_dim])).normal_().to(self.device)
    #    return torch.exp(eps.mul(std).add_(mean))


class BranchIndGammaEncoder(nn.Module):

    def __init__(
            self,
            m_dim, 
            b_prior=torch.tensor([0.1, 0.1]),
            device=torch.device("cpu")):

        super().__init__()

        self.b_dim = 1
        self.m_dim = m_dim
        self.device = device

        # hyper-param for branch mean and sigma LogNormal prior
        self.prior_alpha, self.prior_rate = b_prior

        self.b_alpha = nn.Parameter(torch.zeros(self.m_dim, self.b_dim), requires_grad=True)
        self.b_rate = nn.Parameter(torch.zeros(self.m_dim, self.b_dim), requires_grad=True)
        # To study the type of initialization xavier_uniform_ uniform_ normal_

        nn.init.uniform_(self.b_alpha.data) 
        nn.init.uniform_(self.b_rate.data)

    def forward(self, sample_size):
        
        self.b_alpha.data = F.softplus(self.b_alpha.data)
        self.b_rate.data = F.softplus(self.b_rate.data)
        
#         print("b_alpha")
#         print(self.b_alpha.shape) # [m_dim, b_dim]
#         print(self.b_alpha)
        
        # Prior distribution
        b_dist_p = Gamma(self.prior_alpha, self.prior_rate)

        # Approximate distribution
        b_dist_q = Gamma(self.b_alpha, self.b_rate)
        
        # Sample branch lengths
        b_samples = b_dist_q.rsample(torch.Size([sample_size]))
#         print("b_samples shape {}".format(b_samples.shape)) # [sample_size, m_dim, b_dim]
#         print(b_samples)

        b_kl = kl_divergence(b_dist_q, b_dist_p).view(self.m_dim, 1)
#         print("b_kl")
#         print(b_kl.shape) # [m_dim, 1]
#         print(b_kl)

        return b_samples, b_kl


class BranchIndDeepGammaEncoder(nn.Module):

    def __init__(
            self,
            m_dim,
            h_dim, 
            n_layers=3,
            b_prior=torch.tensor([0.1, 0.1]),
            device=torch.device("cpu")):

        super().__init__()

        self.m_dim = m_dim
        self.h_dim = h_dim
        self.b_dim = 1
        self.n_layers = n_layers
        self.device = device

        # hyper-param for branch mean and sigma LogNormal prior
        self.prior_alpha, self.prior_rate = b_prior
        
        self.noise = torch.zeros((self.m_dim, self.b_dim)).uniform_()
#         self.noise = torch.ones((self.m_dim, self.b_dim)).normal_()

        layers = [nn.Linear(self.b_dim, self.h_dim, bias=True), nn.ReLU()]

        for i in range(1, self.n_layers-1):
            layers.extend([nn.Linear(self.h_dim, self.h_dim, bias=True), nn.ReLU()])

        self.net = nn.Sequential(*layers)

        self.net_alpha = nn.Sequential(
            nn.Linear(self.h_dim, self.b_dim),
            nn.Softplus()) # Sigmoid,

        self.net_rate = nn.Sequential(
            nn.Linear(self.h_dim, self.b_dim),
            nn.Softplus()) 
#             nn.Sigmoid())

    def forward(self, sample_size):

        enc = self.net(self.noise)
        b_alpha = self.net_alpha(enc) #* 10 #.clamp(max=10.)
        b_rate = self.net_rate(enc) #.pow(2) #.clamp(max=100.)

#         print("b_alpha")
#         print(b_alpha.shape) # [m_dim, b_dim]
#         print(b_alpha)

        # Prior distribution
        b_dist_p = Gamma(self.prior_alpha, self.prior_rate)

        # Approximate distribution
        b_dist_q = Gamma(b_alpha, b_rate)

        b_samples = b_dist_q.rsample(torch.Size([sample_size]))
#         print("b_samples shape {}".format(b_samples.shape)) # [sample_size, m_dim, b_dim]
#         print(b_samples)


        b_kl = kl_divergence(b_dist_q, b_dist_p).view(self.m_dim, 1)
#         print("b_kl")
#         print(b_kl.shape) # [m_dim, 1]
#         print(b_kl)

        return b_samples, b_kl


