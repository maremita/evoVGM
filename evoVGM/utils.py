import time
import math
import torch

__author__ = "a.r."

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def get_categorical_prior(conf, prior_type, verbose=False):
    priors = []

    if prior_type in ["ancestor", "freqs"]:
        nb_categories = 4
    elif prior_type == "rates":
        nb_categories = 6
    else:
        raise ValueError("prior type value should be ancestor, freqs or rates")

    if conf == "uniform":
        priors = torch.ones(nb_categories)/nb_categories
    elif "," in conf:
        priors = str2float_tensor(conf, ',', nb_categoriesi, prior_type)
    #elif conf == "empirical": # to be implemented
    #    pass
    else:
        raise ValueError("Check {} prior config values".format(prior_type))

    if verbose:
        print("{} priors: {}".format(prior_type, priors))

    return priors

def get_branch_prior(conf, verbose=False):
    priors = str2float_tensor(conf, ",", 2, "branch")

    if verbose:
        print("Branch priors: {}".format(priors))
    
    return priors 

def str2float_tensor(chaine, sep, nb_values, prior_type):
    values = [float(v) for v in chaine.strip().split(sep)]
    if len(values) != nb_values:
        raise ValueError("the Number of prior values for {} is not correct".format(prior_type))
    return torch.FloatTensor(values)

def str2ints(chaine, sep=","):
    return [int(s) for s in chaine.strip().split(sep)]

def str2floats(chaine, sep=","):
    return [float(s) for s in chaine.strip().split(sep)]

def fasta_to_list(fasta_file, verbose=False):
    # fetch sequences from fasta
    if verbose: print("Fetching sequences from {}".format(fasta_file))
    seqRec_list = SeqCollection.read_bio_file(fasta_file)
    return [seqRec.seq._data for seqRec in seqRec_list] 

