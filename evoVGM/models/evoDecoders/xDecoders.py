import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multinomial import Multinomial

__author__ = "a.r."


class NetXGTRProbDecoder(nn.Module):

    def __init__(self, 
            in_dim,
            h_dim,
            n_layers=3,
            device=torch.device("cpu")):

        super().__init__()

        
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.x_dim = 4  # A, G, C, T
        self.n_layers = n_layers
        self.device = device

        layers = [nn.Linear(self.in_dim, self.h_dim, bias=True), nn.ReLU()]

        for i in range(1, self.n_layers-2):
            layers.extend([nn.Linear(self.h_dim, self.h_dim, bias=True), nn.ReLU()])

#         layers.extend([nn.Linear(self.h_dim, self.x_dim, bias=True), nn.Softmax(-1)])
#         layers.extend([nn.Linear(self.h_dim, self.x_dim, bias=True), nn.LogSoftmax(-1)])
#         layers.extend([nn.Linear(self.h_dim, self.x_dim, bias=True), nn.Softplus()])
        layers.extend([nn.Linear(self.h_dim, self.x_dim, bias=True), nn.Softplus(), nn.Softmax(-1)])

        self.net = nn.Sequential(*layers)

        self.bce_loss = nn.BCELoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')

    # Adpated from https://github.com/zcrabbit/vbpi-nf/blob/main/code/rateMatrix.py#L50
    def buildGTRmatrix(self, rates, pden):
        
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
#         print(beta.shape)
#         print(beta)

        rate_matrix_GTR = torch.zeros((sample_size, 4, 4)).to(self.device)

        for i in range(4):
            for j in range(4):
                if j!=i:
                    rate_matrix_GTR[..., i, j] = pden[..., j]
                    if i+j == 1:
                        rate_matrix_GTR[..., i,j] *= AG
                    if i+j == 2:
                        rate_matrix_GTR[..., i,j] *= AC
                    if i+j == 3 and abs(i-j) > 1:
                        rate_matrix_GTR[..., i,j] *= AT
                    if i+j == 3 and abs(i-j) == 1:
                        rate_matrix_GTR[..., i,j] *= GC
                    if i+j == 4:
                        rate_matrix_GTR[..., i,j] *= GT
                    if i+j == 5:
                        rate_matrix_GTR[..., i,j] *= CT

        for i in range(4):
            rate_matrix_GTR[..., i,i] = - rate_matrix_GTR.sum(dim=-1)[..., i]
        
#         print("\nrate_matrix_GTR")
#         print(rate_matrix_GTR.shape) # [sample_size, x_dim, x_dim]
#         print(rate_matrix_GTR)

        rate_matrix_GTR = torch.einsum("b,bij->bij", (beta, rate_matrix_GTR))
#         print("\nrate_matrix_GTR * beta")
#         print(rate_matrix_GTR.shape) # [sample_size, x_dim, x_dim]
#         print(rate_matrix_GTR)
        
        return rate_matrix_GTR

    def compute_transition_matrix(self, t, r, pi):

#         print("t")
#         print(t.shape) # [sample_size, m_dim, b_dim]
#         print(t)

        rateM = self.buildGTRmatrix(r, pi)
        
#         print("rateM")
#         print(rateM.shape) # [sample_size, x_dim, x_dim]
#         print(rateM)

        transition_matrix = torch.matrix_exp(torch.einsum("bij,bck->bcij", (rateM, t))).clamp(min=0.0, max=1.0)
#         #transition_matrix = torch.einsum("bcij,bjk->bcik", (u_diag, U_inv)).clamp(min=0.0, max=1.0)
#         print("\ntransition_matrix")
#         print(transition_matrix.shape) # [sample_size, m_dim, a_dim, a_dim]
#         print(transition_matrix)

        return transition_matrix

    def forward(self, a, x, transition_matrix):

#         print("a")
#         print(a.shape)  # [sample_size, a_dim]
#         print(a)

#         print("a.unsqueeze(-2)")
#         print(a.unsqueeze(-2).shape)  # [sample_size, 1, a_dim]       
#         print(a.unsqueeze(-2))

#         print('x.shape') # [sample_size, m_dim, x_dim]
#         print(x.shape)
#         print(x)

# # a :  b  i  j     tm:  b  c  j  k
# #     [3, 1, 4]        [3, 2, 4, 4]

#         x_gen = torch.matmul(a, transition_matrix)
        x_gen = torch.einsum("bij,bcjk->bcik", (a.unsqueeze(-2), transition_matrix)).squeeze(-2) #.clamp(min=0., max=1.)
#         print("\nx_gen")
#         print(x_gen.shape) # [sample_size, m_dim, x_dim]
#         print(x_gen)

#         x_gen = F.softmax(x_gen, dim=-1)
#         x_gen = F.normalize(x_gen.abs(), p=1, dim=-1) # 1-norm to normalize the vector

        if x is None:
            ll = None
        else:
            try:
                # Compte the likelihood of each nucleotide of a site
                ll = Multinomial(1, probs=x_gen).log_prob(x).mean(dim=0) #.view(-1, 1)
#                 ll = - self.bce_loss(x_gen, x).mean(-1).mean(dim=0) #..view(-1, 1)

#                 x_recons = self.net(x_gen)

#                 x_recons = self.sample(self.net(x_gen), temperature=0.1)
#                 x_recons = self.sample(x_gen, temperature=0.01)
#                 x_recons = Dirichlet(self.net(x_gen)).rsample()

                x_recons = self.net(self.sample(torch.log(x_gen), temperature=0.1))
#                 x_recons = self.net(Dirichlet(x_gen).rsample())

#                 llx = Multinomial(1, probs=x_recons).log_prob(x).mean(dim=0) #.view(-1, 1)
#                 llx = - self.bce_loss(x_recons, x).mean(-1).mean(dim=0) #.view(-1, 1)
                llx = - self.mse_loss(x_recons, x).mean(-1).mean(dim=0) #.view(-1, 1)
#                 print("llx.shape {}".format(llx.shape))
                
#                 bce = self.mse_loss(x, x_recons)
#                 print("x.shape {}".format(x.shape))
#                 print(x)
#                 print("\nx_recons.shape {}".format(x_recons.shape))
#                 print(x_recons)
#                 print("\nbce.shape {}".format(bce.shape))
#                 print(bce)

            except Exception as e:
                print(e)
                
                ll = -46
                llx = -46
#                 ll = None
#                 raise

#             total_ll = (ll).view(-1, 1)
#             total_ll = (llx).view(-1, 1)
            total_ll = (ll+llx).view(-1, 1)
#             print("\ntotal_ll")
#             print("total_ll shape {}".format(total_ll.shape)) # [m_dim, 1]
#             print(total_ll)

        return x_recons, total_ll

    def sample(self, logits, temperature=1):
        # Reparameterized sampling of discrete distribution
        U = torch.log(torch.rand(logits.shape) + 1e-20)
#         print("U shape {}".format(U.shape))
#         print(U)
        y = logits + U
        y = F.softmax(y/temperature, dim=-1)

        return y
