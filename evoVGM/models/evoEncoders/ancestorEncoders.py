import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.dirichlet import Dirichlet
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence

__author__ = "amine remita"


class AncestorDeepCatEncoder(nn.Module):
    def __init__(
            self,
            in_dim,
            h_dim,
            out_dim, 
            n_layers=3, 
            ancestor_prior=torch.ones(4)/4,
            device=torch.device("cpu")):

        super().__init__()

        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.device_ = device

        # Hyper-param for character prior
        self.pi = ancestor_prior.to(self.device_) 

        layers = [nn.Linear(self.in_dim, self.h_dim,
            bias=True).to(self.device_), nn.ReLU()]

        for i in range(1, self.n_layers-1):
            layers.extend([nn.Linear(self.h_dim, self.h_dim, 
                bias=True).to(self.device_), nn.ReLU()])

        layers.extend([nn.Linear(self.h_dim, self.out_dim,
            bias=True).to(self.device_), nn.LogSoftmax(-1)])

        self.net = nn.Sequential(*layers)

        # Prior distribution
        self.a_dist_p = Categorical(probs=self.pi)

    def forward(self, data, sample_size, a_sample_temp=1):

        data = data.flatten(1)
        a_logit = self.net(data)

        # Sample a
        samples = self.sample(a_logit.expand(
            [sample_size, self.out_dim]), temperature=a_sample_temp)
#         print("samples shape {}".format(samples.shape))
        # [sample_size, a_dim]
#         print(samples)

        # Approximate distribution
        a_dist_q = Categorical(logits=a_logit)

        # KL divergence 
        a_kl = kl_divergence(a_dist_q, self.a_dist_p)
#         print("a_kl")
#         print(a_kl.shape) # [1]
#         print(a_kl)

        return samples, a_kl

    def sample(self, logits, temperature=1):
        # Reparameterized sampling of discrete distribution
        U = torch.log(torch.rand(logits.shape) + 1e-20).to(
                self.device_)
#         print("U shape {}".format(U.shape))
#         print(U)
        y = logits + U
        y = F.softmax(y/temperature, dim=-1)

        return y


class AncestorIndDeepDirEncoder(nn.Module):
    def __init__(
            self,
            a_dim,
            h_dim, 
            n_layers=3, 
            ancestor_prior=torch.ones(4),
            device=torch.device("cpu")):

        super().__init__()

        self.a_dim = a_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.device_ = device

        # Hyper-param for character prior
        self.pi = ancestor_prior

        self.noise = torch.zeros(self.a_dim).uniform_(
                ).to(self.device_) # .normal_()

        layers = [nn.Linear(self.a_dim, self.h_dim,
            bias=True).to(self.device_), nn.ReLU()]

        for i in range(1, self.n_layers-1):
            layers.extend([nn.Linear(self.h_dim, self.h_dim,
                bias=True).to(self.device_), nn.ReLU()])

        layers.extend([nn.Linear(self.h_dim, self.a_dim,
            bias=True).to(self.device_), nn.Softplus()])

        self.net = nn.Sequential(*layers)

        # Prior distribution
        self.a_dist_p = Dirichlet(self.pi)

    def forward(self, sample_size, a_sample_temp=1):

        a_logit = self.net(self.noise)

        # Approximate distribution
        a_dist_q = Dirichlet(a_logit)

        # Sample a
        samples = a_dist_q.rsample(torch.Size([sample_size]))
        #print("samples shape {}".format(samples.shape))
        # [sample_size, a_dim]
        #print(samples)

        # KL divergence 
        a_kl = kl_divergence(a_dist_q, self.a_dist_p)
#         print("a_kl")
#         print(a_kl.shape) # [1]
#         print(a_kl)

        return samples, a_kl


class AncestorDeepDirEncoder(nn.Module):
    def __init__(
            self,
            in_dim,
            h_dim,
            out_dim, 
            n_layers=3, 
            ancestor_prior=torch.ones(4),
            device=torch.device("cpu")):

        super().__init__()

        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.device_ = device

        self.pi = ancestor_prior.to(self.device_) 
        # Hyper-param for character prior

        layers = [nn.Linear(self.in_dim, self.h_dim,
            bias=True).to(self.device_), nn.ReLU()]

        for i in range(1, self.n_layers-1):
            layers.extend([nn.Linear(self.h_dim, self.h_dim,
                bias=True).to(self.device_), nn.ReLU()])

        layers.extend([nn.Linear(self.h_dim, self.out_dim,
            bias=True).to(self.device_), nn.Softplus()])

        self.net = nn.Sequential(*layers)

        # Prior distribution
        self.a_dist_p = Dirichlet(self.pi)

    def forward(self, data, sample_size):

        data = data.squeeze(0).flatten(0)
#         print("data_flatten.shape")
#         print(data.shape)  # [m_dim * x_dim]
        
        a_logit = self.net(data)
#         print("a_logit")
#         print(a_logit.shape) #  torch.Size([a_dim])
#         print(a_logit)

        # Approximate distribution
        a_dist_q = Dirichlet(a_logit)

        # Sample a
        samples = a_dist_q.rsample(torch.Size([sample_size]))
#         print("\nsamples shape {}".format(samples.shape))
        # [sample_size, a_dim]
#         print(samples)

        # KL divergence 
        a_kl = kl_divergence(a_dist_q, self.a_dist_p)
#         print("a_kl")
#         print(a_kl.shape) # [1]
#         print(a_kl)

        return samples, a_kl
