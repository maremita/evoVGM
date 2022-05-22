from evoVGM.models import AncestorDeepCatEncoder
from evoVGM.models import BranchIndDeepGammaEncoder
from evoVGM.models import KappaIndDeepGammaEncoder
from evoVGM.models import XProbDecoder

from evoVGM.utils import timeSince
from evoVGM.models import BaseEvoVGM

import time
import numpy as np
import torch
import torch.nn as nn


__author__ = "amine remita"


class EvoVGM_K80(nn.Module, BaseEvoVGM):
    def __init__(self, 
                 x_dim,
                 a_dim,
                 h_dim,
                 m_dim,
                 nb_layers=3,
                 ancestor_prior=torch.ones(4)/4,
                 branch_prior=torch.tensor([0.1, 0.1]),
                 kappa_prior=torch.tensor([0.1]),
                 device=torch.device("cpu")):

        super().__init__()

        self.x_dim = x_dim
        self.a_dim = a_dim
        self.h_dim = h_dim
        self.m_dim = m_dim
        self.nb_layers = nb_layers
        # hyper priors
        self.ancestor_prior = ancestor_prior
        self.branch_prior = branch_prior
        self.kappa_prior = kappa_prior
        #
        self.device = device

        self.freqs = (torch.ones(4)/4).reshape(1, -1)

        # Ancestor encoder
        self.ancestEncoder = AncestorDeepCatEncoder(
                self.x_dim*self.m_dim,
                self.h_dim, 
                self.a_dim, 
                n_layers=self.nb_layers, 
                ancestor_prior=self.ancestor_prior,
                device=self.device)

        # Branche encoder  
        self.branchEncoder = BranchIndDeepGammaEncoder(
                self.m_dim, 
                self.h_dim,
                n_layers=self.nb_layers,
                b_prior=self.branch_prior,
                device=self.device)

        # Branche encoder  
        self.kappaEncoder = KappaIndDeepGammaEncoder(
                self.h_dim,
                n_layers=self.nb_layers,
                k_prior=self.kappa_prior,
                device=self.device)

        # decoder
        self.decoder = XProbDecoder(
                device=self.device)

    def forward(self, 
            sites, 
            site_counts,
            latent_sample_size=10,
            sample_temp=0.1,
            alpha_kl=0.001,
            shuffle_sites=True,
            keep_vars=False):

        sites_size, nb_seqs, feat_size = sites.shape

        assert(self.x_dim == feat_size)
        assert(self.m_dim == nb_seqs)

        ancestors = torch.tensor([]).to(self.device).detach()
        branches = torch.tensor([]).to(self.device).detach()
        kappas = torch.tensor([]).to(self.device).detach()
        x_recons = torch.tensor([]).to(self.device).detach()

        if keep_vars:
            ancestors_var = torch.tensor([]).to(self.device).detach()
            branches_var = torch.tensor([]).to(self.device).detach()
            kappas_var = torch.tensor([]).to(self.device).detach()
            x_recons_var = torch.tensor([]).to(self.device).detach()

        alpha_kl = torch.tensor(alpha_kl).to(self.device)

        #logl = torch.zeros(self.m_dim, 1).to(
        #        self.device).detach()
        logl = torch.zeros(1).to(self.device).detach()
        a_kl_ws = torch.zeros(1).to(self.device).detach()
        kl_qprior = torch.zeros(1).to(self.device).detach()
        elbo = torch.zeros(1).to(self.device)

        N = site_counts.sum().detach()
 
        # Sample Branche lengths
        b_ws, b_kl_ws = self.branchEncoder(latent_sample_size)
        # ws = whole sequence
        #print("b_kl_ws")
        #print(b_kl_ws.shape) # [m_dim, 1]
        #print(b_kl_ws)
        
        # Sample Kappa
        k_ws, k_kl_ws = self.kappaEncoder(latent_sample_size)
        #print("k_kl_ws")
        #print(k_kl_ws.shape) # [1]
        #print(k_kl_ws)

        # update rates
        rates_k = self.decoder.compute_rates_kappa(k_ws)

        kl_qprior = N * (b_kl_ws.sum(0) + k_kl_ws)
        #print("kl_qprior")
        #print(kl_qprior.shape) # [1]
        #print(kl_qprior)
        
        # Compute the transition probabilities matrices
        tm = self.decoder.compute_transition_matrix(b_ws, 
                rates_k, self.freqs)

        with torch.no_grad():
            branches = b_ws.mean(0, keepdim=True)
            kappas = k_ws.mean(0, keepdim=True)

            if keep_vars:
                branches_var = b_ws.var(0, keepdim=True)
                kappas_var = k_ws.var(0, keepdim=True)
 
        # shuffling indices
        indices = [i for i in range(sites_size)]

        if shuffle_sites :
            np.random.shuffle(indices)
        
        for n in indices:
            x_n = sites[n, :, :].unsqueeze(0)
            x_n_expanded = x_n.expand([latent_sample_size,
                self.m_dim, self.x_dim])
            #print('\nx_n_expanded') # [sample_size, m_dim, x_dim]
            #print(x_n_expanded.shape) # [sample_size, m_dim, x_dim]
            #print(x_n_expanded)
            #print()

            # ANCESTOR Encoder 
            # ################
            #a_n, a_kl_n =self.ancestEncoder(x_n, latent_sample_size)
            a_n, a_kl_n = self.ancestEncoder(x_n, latent_sample_size,
                    a_sample_temp=sample_temp)
            #a_n, a_kl_n = self.ancestEncoder(latent_sample_size)

            a_kl_ws += a_kl_n * site_counts[n]

            # X decoder and Log likelihood
            # ############################
            x_recons_n, loglx_n = self.decoder(a_n, x_n_expanded,
                    tm, self.freqs)
            #print("loglx_n.shape {} ".format(loglx_n.shape))
            # [m_dim, 1]
            #print(loglx_n)

            logl += loglx_n * site_counts[n]

            with torch.no_grad():
                ancestors = torch.cat(
                        [ancestors, a_n.mean(0, keepdim=True)], 0)
                x_recons = torch.cat(
                        [x_recons, x_recons_n.mean(0, keepdim=True)],
                        0)

                if keep_vars:
                    ancestors_var = torch.cat(
                            [ancestors_var,
                                a_n.var(0, keepdim=True)], 0)
                    x_recons_var = torch.cat(
                            [x_recons_var, x_recons_n.var(0,
                                keepdim=True)], 0)
        #print("logl")
        #print(logl.shape) # [1]
        #print(logl)
 
        #print("kl_qprior")
        #print(kl_qprior.shape) # [1]
        #print(kl_qprior)

        # Compute ELBO
        ########################
        kl_qprior += a_kl_ws
        #elbo += ( logl - (alpha_kl * kl_qprior)).sum(0)
        elbo += ( logl - (alpha_kl * kl_qprior))
        #print("elbo")
        #print(elbo.shape) # [1]
        #print(elbo)

        ret_values = dict(
                elbo=elbo,
                logl=logl,
                kl_qprior=kl_qprior,
                a=ancestors,
                b=branches,
                k=kappas,
                x=x_recons)

        if keep_vars:
            ret_values["a_var"] = ancestors_var
            ret_values["b_var"] = branches_var 
            ret_values["k_var"] = kappas_var 
            ret_values["x_var"] = x_recons_var
 
        return ret_values
