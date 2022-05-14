import torch
from torch.distributions.multinomial import Multinomial

__author__ = "amine remita"


def compute_log_likelihood_data(
        data,
        data_counts,
        branches,
        gtrrates,
        gtrfreqs):

    """
    Compute the log likelihood of the data knowing the true paramters.
    The logl is computed using the formula from evoVGM model (top-down)
    """

    if isinstance(branches, str):
        branch_lengths = torch.tensor(
                [float(v) for v in branches.split(",")]).view(-1, 1)
    elif isinstance(branches, list):
        branch_lengths = torch.tensor(branches)
    elif isinstance(branches, torch.Tensor):
        branch_lengths = branches
     
    if isinstance(gtrrates, list):
        gtrrates = torch.tensor(gtrrates)

    if isinstance(gtrfreqs, list):
        gtrfreqs = torch.tensor(gtrfreqs)

    def buildmatrix(rates, pden):

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

        beta = (1.0/(2*(AG*pA*pG+AC*pA*pC+AT*pA*pT+GC*pG*pC+GT*pG*pT+CT*pC*pT)))

        rate_matrix_GTR = torch.zeros((4, 4))

        for i in range(4):
            for j in range(4):
                if j!=i:
                    rate_matrix_GTR[..., i,j] = pden[...,j]
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

        rate_matrix_GTR = beta * rate_matrix_GTR

        return rate_matrix_GTR

    def compute_transition_matrix(t, r, pi):

        rateM = buildmatrix(r, pi)

        transition_matrix = torch.matrix_exp(
                torch.einsum("ij,ck->cij", (rateM, t))).clamp(min=0.0, max=1.0)

        return transition_matrix

    # Algorithm
    with torch.no_grad():
        #
        trans_mats = compute_transition_matrix(branch_lengths, gtrrates, gtrfreqs)

        sites_size = data.shape[0]

        indices = [i for i in range(sites_size)]

        logl = 0

        for n in indices:

            a_n = data[n, 0, :]
            x_n = data[n, 1:, :]

            log_p_a_n = torch.matmul(gtrfreqs, a_n).log()

            x_gen = torch.matmul(a_n, trans_mats) # Get prob transitions of nucleotide a_n

            log_p_x_a_n = Multinomial(1, probs=x_gen).log_prob(x_n).sum()

            logl += (log_p_x_a_n) * data_counts[n]

    return logl
