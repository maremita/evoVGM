import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multinomial import Multinomial

__author__ = "amine remita"


class XProbDecoder(nn.Module):

    def __init__(self,
            device=torch.device("cpu")):
        super().__init__()

        self.device = device

    def buildmatrix(self, rates, pden):
    # Adpated from https://github.com/zcrabbit/vbpi-nf/blob/main/code/rateMatrix.py#L50

        sample_size = rates.shape[0]

        pA = pden[...,0]
        pG = pden[...,1]
        pC = pden[...,2]
        pT = pden[...,3]

        AG = rates[...,0]
        AC = rates[...,1]
        AT = rates[...,2]
        GC = rates[...,3]
        GT = rates[...,4]
        CT = rates[...,5]

#         print("pA: {}".format(pA))
#         print("pG: {}".format(pG))
#         print("pC: {}".format(pC))
#         print("pT: {}".format(pT))
#         print()

#         print("AG: {}".format(AG))
#         print("AC: {}".format(AC))
#         print("AT: {}".format(AT))
#         print("GC: {}".format(GC))
#         print("GT: {}".format(GT))
#         print("CT: {}".format(CT))

        beta = (1.0/(2*(AG*pA*pG+AC*pA*pC+AT*pA*pT+GC*pG*pC+GT*pG*pT+CT*pC*pT)))
#         print("\nbeta")
#         print(beta.shape) #[sample_size]
#         print(beta)

        rate_matrix = torch.zeros((sample_size, 4, 4)).to(self.device)

        for i in range(4):
            for j in range(4):
                if j!=i:
#                     rate_matrix[:,:, i,j] = pden[j]
                    rate_matrix[..., i,j] = pden[...,j]
                    if i+j == 1:
                        rate_matrix[..., i,j] *= AG
                    if i+j == 2:
                        rate_matrix[..., i,j] *= AC
                    if i+j == 3 and abs(i-j) > 1:
                        rate_matrix[..., i,j] *= AT
                    if i+j == 3 and abs(i-j) == 1:
                        rate_matrix[..., i,j] *= GC
                    if i+j == 4:
                        rate_matrix[..., i,j] *= GT
                    if i+j == 5:
                        rate_matrix[..., i,j] *= CT

        for i in range(4):
            rate_matrix[..., i,i] = - rate_matrix.sum(dim=-1)[..., i]

#         print("\nrate_matrix")
#         print(rate_matrix.shape) # [sample_size, x_dim, x_dim]
#         print(rate_matrix)

        rate_matrix = torch.einsum("b,bij->bij", (beta, rate_matrix))
#         print("\nrate_matrix * beta")
#         print(rate_matrix.shape) # [sample_size, x_dim, x_dim]
#         print(rate_matrix)

        return rate_matrix

    def compute_transition_matrix(self, t, r, pi):

#         print("t")
#         print(t.shape) # [sample_size, m_dim, b_dim]
#         print(t)

        rateM = self.buildmatrix(r, pi)

#         print("rateM")
#         print(rateM.shape) # [sample_size, x_dim, x_dim]
#         print(rateM)

        transition_matrix = torch.matrix_exp(torch.einsum("bij,bck->bcij", (rateM, t))).clamp(min=0.0, max=1.0)
#         #transition_matrix = torch.einsum("bcij,bjk->bcik", (u_diag, U_inv)).clamp(min=0.0, max=1.0)
#         print("\ntransition_matrix")
#         print(transition_matrix.shape) # [sample_size, m_dim, a_dim, a_dim]
#         print(transition_matrix)

        return transition_matrix

    def forward(self, a, x, transition_matrix, pi):

#         print("pi")
#         print(pi.shape)  # [sample_size, a_dim]
#         print(pi)

#         print("a")
#         print(a.shape)  # [sample_size, a_dim]
#         print(a)

#         log_pi_a = torch.matmul(pi.unsqueeze(-2), a.unsqueeze(-1)).log().view(-1, 1) #.mean(dim=0)  #####$
#         print("log_pi_a")
#         print(log_pi_a.shape)  # [sample_size, 1]
#         print(log_pi_a)


#         print("a.unsqueeze(-2)")
#         print(a.unsqueeze(-2).shape)  # [sample_size, 1, a_dim]
#         print(a.unsqueeze(-2))

#         print('x.shape') # [sample_size, m_dim, x_dim]
#         print(x.shape)
#         print(x)

# # a :  b  i  j     tm:  b  c  j  k
# #     [3, 1, 4]        [3, 2, 4, 4]

#         x_recons = torch.matmul(a, transition_matrix)
        x_gen = torch.einsum("bij,bcjk->bcik", (a.unsqueeze(-2), transition_matrix)).squeeze(-2) #.clamp(min=0., max=1.)
#         print("\nx_gen")
#         print(x_gen.shape) # [sample_size, m_dim, x_dim]
#         print(x_gen)

#         x_gen = F.softmax(x_gen, dim=-1)
#         x_gen = F.normalize(x_gen.abs(), p=1, dim=-1) # 1-norm to normalize the vector


#         log_pi_x = torch.einsum("bij,bij->bi", (pi.unsqueeze(-2), x_gen)).log().mean(dim=0).view(-1, 1) # faire un dot product #.mean(dim=0)
#         print("log_pi_x")
#         print(log_pi_x.shape)  # [1, 1]
#         print(log_pi_x)

        if x is None:
            ll = None
        else:
            try:
                # Compte the likelihood of each nucleotide of a site
                #log_p_x_a = Multinomial(1, probs=x_gen).log_prob(x).mean(dim=0).view(-1, 1)    #####
                log_p_x_a = Multinomial(1, probs=x_gen).log_prob(x).sum(dim=1).view(-1, 1)  #####$
#                 log_p_x_a = Multinomial(1, probs=x_gen).log_prob(x) #.view(-1, 1)  # .mean(dim=0)
#                 ll = - self.bce_loss(x_gen, x).mean(-1).mean(dim=0).view(-1, 1)

#                 print("log_p_x_a")
#                 print(log_p_x_a.shape)
#                 print(log_p_x_a)

            except Exception as e:
                print(e)
                log_p_x_a = -46
#                 ll = None
#                 raise

#         ll = (log_pi_a + log_p_x_a).mean(0)  #####$
        ll = (log_p_x_a).mean(0)
#         ll = log_pi_x + log_p_x_a
        #ll = log_p_x_a

#         print("\nll")
#         print("ll shape {}".format(ll.shape)) # [m_dim, 1]
#         print(ll)

        return x_gen, ll

    def compute_rates_kappa(self, kappa):
        """
        "AG", "AC", "AT", "GC", "GT", "CT"
        Multiply AG and CT transition rates by kappa
        """
        nb_sample = kappa.shape[0]

        rates = torch.hstack((
            kappa*1/6, 
            (torch.ones(4)/6).expand(nb_sample, -1), 
            kappa*1/6))

        return rates/rates.sum(1, keepdim=True)
