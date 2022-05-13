#!/usr/bin/env python

from evoVGM.simulations import evolve_seqs_full_homogeneity 
from evoVGM.simulations import build_star_tree
from evoVGM.data import build_msa_categorical
from evoVGM.utils import timeSince, get_categorical_prior, get_branch_prior 
from evoVGM.models import EvoVGM_GTR 

import sys
import os
import configparser
import time
import random

import numpy as np
import torch

torch.set_printoptions(precision=4, sci_mode=False)


# Get argument values from ini file
config_file = sys.argv[1]
config = configparser.ConfigParser(
        interpolation=configparser.ExtendedInterpolation())

with open(config_file, "r") as cf:
    config.read_file(cf)

# seeting parameters
seed = config.getint("settings", "seed")
device_str = config.get("settings", "device")
verbose = config.getboolean("settings", "verbose")
deterministic = config.getboolean("settings", "deterministic")

# Hyper parameters
h_dim = config.getint("hperparams", "hidden_size")
nb_samples = config.getint("hperparams", "nb_samples")
sample_temp = config.getfloat("hperparams", "sample_temp")
alpha_kl = config.getfloat("hperparams", "alpha_kl")
n_epochs = config.getint("hperparams", "n_epochs")
learning_rate = config.getfloat("hperparams", "learning_rate")
weight_decay = config.getfloat("hperparams", "optim_weight_decay")
optim = config.get("hperparams", "optim")

# priors values
ancestor_prior_conf = config.get("priors", "ancestor_prior")
branch_prior_conf = config.get("priors", "branch_prior") 
rates_prior_conf = config.get("priors", "rates_prior")
freqs_prior_conf = config.get("priors", "freqs_prior")


# Computing device setting
if device_str != "cpu" and not torch.cuda.is_available():
    if verbose: print("Cuda is not available. Changing device to 'cpu'")
    device_str = 'cpu'

device = torch.device(device_str)

# Reproducibility: set seed of randomness
if deterministic:
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if device.type == "cuda":
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
        # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"


# Data preparation
alignment_len = 10000
b_str = "0.12,0.29,0.45,0.14"

#            "AG"  "AC"  "AT"  "GC"  "GT" "CT"
sim_rates = [0.16, 0.05, 0.16, 0.09, 0.3, 0.24]

#             A     C    G     T
sim_freqs = [0.1, 0.45, 0.3, 0.15]

tree=build_star_tree(b_str)

ancestor, sequences = evolve_seqs_full_homogeneity(
        tree,
        fasta_file=None, 
        nb_sites=alignment_len,
        subst_rates=sim_rates,
        state_freqs=sim_freqs,
        return_anc=True,
        verbose=verbose)

motifs_cats = build_msa_categorical(sequences)
X = torch.from_numpy(motifs_cats.data).to(device)
X, X_counts = X.unique(dim=0, return_counts=True)

x_dim = 4
a_dim = 4
m_dim = len(sequences) # Number of sequences

print_every = 10

# Get prior values
ancestor_prior = get_categorical_prior(ancestor_prior_conf,
        "ancestor", verbose=verbose)
rates_prior = get_categorical_prior(rates_prior_conf, "rates",
        verbose=verbose)
freqs_prior = get_categorical_prior(freqs_prior_conf, "freqs",
        verbose=verbose)
branch_prior = get_branch_prior(branch_prior_conf, verbose=verbose)

# Instanciate the model

# EvoVGTRW_KL infer ancestor and branch length and GTR params latent variables
evoModel = EvoVGM_GTR(x_dim, a_dim, h_dim, m_dim,
        ancestor_prior=ancestor_prior,
        branch_prior=branch_prior,
        rates_prior=rates_prior,
        freqs_prior=freqs_prior,
        device=device).to(device)

#print("\nancestor encoder device is {}\n".format(evoModel.ancestEncoder.encoder[0].weight.type()))

# SGD Optimizer
if optim == "adam":
    optimizer = torch.optim.Adam(evoModel.parameters(), 
            lr=learning_rate, weight_decay=weight_decay)
else:
    optimizer = torch.optim.SGD(evoModel.parameters(),
            lr=learning_rate, weight_decay=weight_decay)

start = time.time()
n_dim = X_counts.sum()

for epoch in range(1, n_epochs + 1):

    optimizer.zero_grad()
    try:
        elbos, lls, kls, ancestors, branches, gtrrates, gtrfreqs, xrecons = evoModel(X, X_counts, nb_samples, sample_temp, alpha_kl)
        loss = - elbos
        loss.backward()
        optimizer.step()
    except Exception as e:
        """Stop the training if there is a problem with backward pass"""
        print("\nStopping training at epoch {}  because of an exception".format(epoch))
        print(e)
        break

    # printing
    with torch.no_grad():
        if epoch % print_every == 0 or epoch <= 10:
            print("{}\t Train Epoch: {} \t ELBO: {:.3f}\t Lls {:.3f}\t KLs {:.3f}".format(
                timeSince(start), epoch, elbos.item()/n_dim, lls.item()/n_dim, kls.item()/n_dim), end="\r")

