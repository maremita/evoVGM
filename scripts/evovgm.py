#!/usr/bin/env python

from evoVGM.simulations import evolve_seqs_full_homogeneity as evolve_sequences
from evoVGM.simulations import build_star_tree

from evoVGM.data import build_msa_categorical

from evoVGM.utils import timeSince
from evoVGM.utils import get_categorical_prior 
from evoVGM.utils import get_branch_prior 
from evoVGM.utils import get_kappa_prior 
from evoVGM.utils import str2floats, fasta_to_list
from evoVGM.utils import str_to_values
from evoVGM.utils import write_conf_packages

from evoVGM.reports import plt_elbo_ll_kl_rep_figure
from evoVGM.reports import aggregate_estimate_values
from evoVGM.reports import plot_fit_estim_dist
from evoVGM.reports import plot_fit_estim_corr
from evoVGM.reports import plot_fit_seq_dist
from evoVGM.reports import aggregate_sampled_estimates
from evoVGM.reports import report_sampled_estimates

from evoVGM.models import compute_log_likelihood_data 
from evoVGM.models import EvoVGM_JC69
from evoVGM.models import EvoVGM_K80
from evoVGM.models import EvoVGM_GTR 

import sys
import os
from os import makedirs
import configparser
import time
from datetime import datetime

import numpy as np
import torch

from joblib import Parallel, delayed
from joblib import dump, load


__author__ = "amine remita"


"""
evovgm.py is the main program that uses the EvoVGM 
model with different substitution models (JC69, K80 and GTR).
The experiment of fitting the model can be done for
<nb_replicates> times.
The program can use a sequence alignment from a Fasta file or
simulate a new sequence alignment using evolutionary parameters
defined in the config file.

Once the evoVGM package is installed, you can run evovgm.py 
using this command line:

# evovgm.py evovgm_conf_template.ini
where <evovgm_conf_template.ini> is the config file.
"""

## Evaluation function
## ###################
def eval_evomodel(EvoModel, m_args, fit_args):
    overall = dict()

    # Instanciate the model
    e = EvoModel(**m_args)

    ## Fitting and param3ter estimation
    ## ################################
    ret = e.fit(fit_args["X"], fit_args["X_counts"],
            latent_sample_size=fit_args["nb_samples"],
            sample_temp=fit_args["sample_temp"], 
            alpha_kl=fit_args["alpha_kl"], 
            max_iter=fit_args["n_epochs"],
            optim=fit_args["optim"], 
            optim_learning_rate=fit_args["learning_rate"],
            optim_weight_decay=fit_args["weight_decay"], 
            X_val=fit_args["X_val"],
            X_val_counts=fit_args["X_val_counts"],
            A_val=fit_args["A_val"],
            keep_val_history=fit_args["keep_val_history"],
            keep_fit_history=fit_args["keep_fit_history"],
            verbose=fit_args["verbose"]
            )

    fit_hist = [ret["elbos_list"], ret["lls_list"], ret["kls_list"]]

    if fit_args["X_val"] is not None:
        fit_hist.extend([
            ret["elbos_val_list"],
            ret["lls_val_list"],
            ret["kls_val_list"]])

        overall["val_hist_estim"] = ret["val_estimates"]
    else:
        overall["fit_hist_estim"] = ret["fit_estimates"]

    overall["fit_hist"] = np.array(fit_hist)

    ## Generation after fitting
    ## ########################
    # If Validation is False, X_val corresponds to X.
    # the key in oldest versions was "val_results"
    overall["gen_results"] = e.generate(
            # X_gen here corresponds to X_val if validation is True,
            # otherwise it corresponds to X.
            fit_args["X_gen"],
            None,
            latent_sample_size=fit_args["nb_samples"],
            sample_temp=fit_args["sample_temp"], 
            alpha_kl=fit_args["alpha_kl"],
            keep_vars=fit_args["keep_gen_vars"])

    return overall


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Config file is missing!!")
        sys.exit()

    print("\nRunning {}".format(sys.argv[0]), flush=True)

    ## Fetch argument values from ini file
    ## ###################################
    config_file = sys.argv[1]
    config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation())

    with open(config_file, "r") as cf:
        config.read_file(cf)

    # IO files
    output_path = config.get("io", "output_path")
    scores_from_file = config.getboolean("io", "scores_from_file",
            fallback=False)

    # Sequence data
    validation = config.getboolean("data", "validation",
            fallback=True)
    sim_data = config.getboolean("data", "sim_data",
            fallback=False)
    sim_from_fasta = config.getboolean("data", "sim_from_fasta",
            fallback=False)
    fasta_fit_file = config.get("data", "fasta_fit_file",
            fallback="")
    fasta_val_file = config.get("data", "fasta_val_file",
            fallback="")
    nb_sites = config.getint("data", "alignment_size", fallback=100)
    sim_blengths = config.get("data", "branch_lengths",
            fallback="0.1,0.1")
    sim_rates = str_to_values(config.get("data", "rates",
        fallback="0.16"), 6, cast=float)
    sim_freqs = str_to_values(config.get("data", "freqs",
        fallback="0.25"), 4, cast=float)

    # The order of freqs is different for evoVGM
    # A G C T
    sim_freqs_vgm = [sim_freqs[0], sim_freqs[2],
            sim_freqs[1], sim_freqs[3]]

    # setting parameters
    job_name = config.get("settings", "job_name", fallback=None)
    seed = config.getint("settings", "seed",
            fallback=42)
    device_str = config.get("settings", "device",
            fallback="cpu")
    verbose = config.get("settings", "verbose",
            fallback=1)
    compress_files = config.getboolean("settings", "compress_files",
            fallback=False)

    # Evo variational model type
    evomodel_type = config.get("vb_model", "evomodel", # subvmodel
            fallback="gtr")

    # Hyper parameters
    nb_replicates = config.getint("hyperparams", "nb_replicates",
            fallback=2)
    nb_samples = config.getint("hyperparams", "nb_samples",
            fallback=10)
    h_dim = config.getint("hyperparams", "hidden_size",
            fallback=32)
    nb_layers = config.getint("hyperparams", "nb_layers", fallback=3)
    sample_temp = config.getfloat("hyperparams", "sample_temp",
            fallback=0.1)
    alpha_kl = config.getfloat("hyperparams", "alpha_kl",
            fallback=0.0001)
    n_epochs = config.getint("hyperparams", "n_epochs",
            fallback=100)
    optim = config.get("hyperparams", "optim",
            fallback="adam")
    learning_rate = config.getfloat("hyperparams", "learning_rate",
            fallback=0.005)
    weight_decay = config.getfloat("hyperparams",
            "optim_weight_decay", fallback=0.00001)

    # Hyper-parameters of prior distributions
    ancestor_hp_conf = config.get("priors", "ancestor_prior_hp", 
            fallback="uniform")
    branch_hp_conf = config.get("priors", "branch_prior_hp",
            fallback="0.1,0.1")
    kappa_hp_conf = config.get("priors", "kappa_prior_hp",
            fallback="1.,1.")
    rates_hp_conf = config.get("priors", "rates_prior_hp",
            fallback="uniform")
    freqs_hp_conf = config.get("priors", "freqs_prior_hp",
            fallback="uniform")

    # plotting settings
    size_font = config.getint("plotting", "size_font", fallback=16)
    plt_usetex = config.getboolean("plotting", "plt_usetex",
            fallback=False)
    print_xtick_every = config.getint("plotting",
            "print_xtick_every", fallback=10)

    # Process verbose
    if verbose.lower() == "false":
        verbose = 0
    elif verbose.lower() == "none":
        verbose = 0
    elif verbose.lower() == "true":
        verbose = 1
    else:
        try:
            verbose = int(verbose)
            if verbose < 0:
                print("\nInvalid value for verbose"\
                        " {}".format(verbose))
                print("Valid values are: True, False, None and"\
                        " positive integers")
                print("Verbose is set to 0")
                verbose = 0
        except ValueError as e:
            print("\nInvalid value for verbose {}".format(verbose))
            print("Valid values are: True, False, None and"\
                    " positive integers")
            print("Verbose is set to 1")
            verbose = 1

    if evomodel_type not in ["jc69", "k80", "gtr"]:
        print("\nevomodel_type should be jc69, k80 or gtr,"\
                " not {}".format(evomodel_type), file=sys.stderr)
        sys.exit()

    # Computing device setting
    if device_str != "cpu" and not torch.cuda.is_available():
        if verbose: 
            print("\nCuda is not available."\
                    " Changing device to 'cpu'")
        device_str = 'cpu'

    device = torch.device(device_str)

    if str(job_name).lower() in ["auto", "none"]:
        job_name = None

    if not job_name:
        now = datetime.now()
        job_name = now.strftime("%y%m%d%H%M")

    sim_params = dict(
            b=np.array(str2floats(sim_blengths)),
            r=np.array(sim_rates),
            f=np.array(sim_freqs_vgm),
            k=np.array([[sim_rates[0]/sim_rates[1]]])
            )

    ## output name file
    ## ################
    output_path = os.path.join(output_path, evomodel_type, job_name)
    makedirs(output_path, mode=0o700, exist_ok=True)

    if verbose:
        print("\nExperiment output: {}".format(
            output_path))

    ## Loading results from file
    ## #########################
    results_file = output_path+"/{}_results.pkl".format(job_name)

    if os.path.isfile(results_file) and scores_from_file:
        if verbose: print("\nLoading scores from file...")
        result_data = load(results_file)
        rep_results=result_data["rep_results"]

    ## Execute the evaluation and save results
    ## #######################################
    else:
        if verbose: print("\nRunning the evaluation...")

        if verbose:
            print("\nValidation during fitting: {}".format(
                validation))

        # Validation related variables
        A_val = None
        X_val = None
        X_val_counts = None

        ## Data preparation
        ## ################
        if sim_data:
            # Files paths of simulated data
            # training sequences
            x_fasta_file = output_path+"/{}_train.fasta".format(
                    job_name)
            # validation sequences
            v_fasta_file = output_path+"/{}_valid.fasta".format(
                    job_name)

            # Extract data from simulated FASTA files if they exist
            if os.path.isfile(x_fasta_file) and sim_from_fasta:
                if verbose: 
                    print("\nLoading simulated sequences"\
                            " from files...")
                # Load from files
                # Here, simulated FASTA file contain the
                # root sequence
                ax_sequences = fasta_to_list(x_fasta_file, verbose)
                x_root = ax_sequences[0]
                x_sequences = ax_sequences[1:]

                if validation and os.path.isfile(v_fasta_file):
                    av_sequences = fasta_to_list(v_fasta_file,
                            verbose)
                    v_root = av_sequences[0]
                    v_sequences = av_sequences[1:]

            # Simulate new data
            else:
                if verbose: print("\nSimulating sequences...")
                # Evolve sequences
                tree=build_star_tree(sim_blengths)

                x_root, x_sequences = evolve_sequences(
                        tree,
                        fasta_file=x_fasta_file,
                        nb_sites=nb_sites,
                        subst_rates=sim_rates,
                        state_freqs=sim_freqs,
                        return_anc=True,
                        seed=seed,
                        verbose=verbose)

                if validation:
                    v_root, v_sequences = evolve_sequences(
                            tree,
                            fasta_file=v_fasta_file,
                            nb_sites=nb_sites,
                            subst_rates=sim_rates,
                            state_freqs=sim_freqs,
                            return_anc=True,
                            seed=seed+42,
                            verbose=verbose)

            # Transform the fitting sequences including ancestral
            # sequence. This will be used to compute the true logl
            # knowing the true parameters
            all_x_seqs = [x_root, *x_sequences]
            AX_unique, AX_counts = torch.from_numpy(
                    build_msa_categorical(all_x_seqs).data).unique(
                    dim=0, return_counts=True)

            x_logl_data = compute_log_likelihood_data(AX_unique,
                    AX_counts, sim_blengths, sim_rates, 
                    sim_freqs_vgm)

            if verbose:
                print("\nLog likelihood of the fitting data"\
                        " {}".format(x_logl_data))

            if validation:
                # Transform Val ancestral sequence (np.ndarray)
                A_val = build_msa_categorical(
                        v_root).data.reshape(-1, 4)

                # Transform the validation sequences including 
                # ancestral sequence. this will be used to compute
                # the true logl knowing the true parameters
                all_v_seqs = [v_root, *v_sequences]
                AV_unique, AV_counts = torch.from_numpy(
                        build_msa_categorical(
                            all_v_seqs).data).unique(dim=0, 
                                    return_counts=True)

                v_logl_data = compute_log_likelihood_data(AV_unique,
                        AV_counts, sim_blengths, sim_rates, 
                        sim_freqs_vgm)

                if verbose:
                    print("Log likelihood of the validation data"\
                            " {}".format(v_logl_data))

        # Extract data from given FASTA files
        else:
            # Files paths of given FASTA files
            # training sequences
            x_fasta_file = fasta_fit_file
            # validation sequences
            v_fasta_file = fasta_fit_file

            if validation and os.path.isfile(fasta_val_file):
                v_fasta_file = fasta_val_file

            # Given FASTA files do not contain root sequences
            if verbose: print("\nLoading sequences from files...")
            x_sequences = fasta_to_list(x_fasta_file, verbose)

            if validation:
                v_sequences = fasta_to_list(v_fasta_file, verbose)
        # End of fetching/simulating the data

        # Update file paths in config file
        config.set("data", "fasta_fit_file", x_fasta_file)
        if validation:
            config.set("data", "fasta_val_file", v_fasta_file)
        else:
            config.remove_option("data", "fasta_val_file")

        # Transform fitting sequences
        x_motifs_cats = build_msa_categorical(x_sequences)
        X = torch.from_numpy(x_motifs_cats.data).to(device)
        X_gen = X.clone().detach() # will be used in generation
        X, X_counts = X.unique(dim=0, return_counts=True)

        if validation:
            # Transform validation sequences
            # No need to get unique sites.
            # Validation need only forward pass and it's fast.
            v_motifs_cats = build_msa_categorical(v_sequences)
            X_val = torch.from_numpy(v_motifs_cats.data).to(device)
            # X_gen here corresponds to X_val if validation is True,
            # otherwise it corresponds to X.
            X_gen = X_val.clone().detach() # will be used in gen.
            X_val_counts = None 

        # Set dimensions
        x_dim = 4
        a_dim = 4
        m_dim = len(x_sequences) # Number of sequences

        ## Get prior hyper-parameter values
        ## ################################
        if verbose:
            print("\nGet the prior hyper-parameters...")

        ancestor_prior_hp = get_categorical_prior(ancestor_hp_conf,
                "ancestor", verbose=verbose)
        branch_prior_hp = get_branch_prior(branch_hp_conf,
                verbose=verbose)

        if evomodel_type == "gtr":
            # Get rate and freq priors if the model is EvoVGM_GTR
            rates_prior_hp = get_categorical_prior(rates_hp_conf, 
                    "rates", verbose=verbose)
            freqs_prior_hp = get_categorical_prior(freqs_hp_conf,
                    "freqs", verbose=verbose)

        elif evomodel_type == "k80":
            kappa_prior_hp = get_kappa_prior(kappa_hp_conf, 
                    verbose=verbose)

        if verbose: print()
        ## Evo model type
        ## ##############
        if evomodel_type == "gtr":
            EvoModelClass = EvoVGM_GTR
        elif evomodel_type == "k80":
            EvoModelClass = EvoVGM_K80
        else:
            EvoModelClass = EvoVGM_JC69

        model_args = {
                "x_dim":x_dim,
                "a_dim":a_dim,
                "h_dim":h_dim,
                "m_dim":m_dim,
                "ancestor_prior_hp":ancestor_prior_hp,
                "branch_prior_hp":branch_prior_hp,
                "device":device
                }

        if evomodel_type == "gtr":
            # Add rate and freq priors if the model is EvoVGM_GTR
            model_args["rates_prior_hp"] = rates_prior_hp
            model_args["freqs_prior_hp"] = freqs_prior_hp

        elif evomodel_type == "k80":
            model_args["kappa_prior_hp"] = kappa_prior_hp

        if validation:
            keep_fit_history=False
            keep_val_history=True
        else:
            keep_fit_history=True
            keep_val_history=False

        # Fitting the parameters
        fit_args = {
                "X":X,
                "X_counts":X_counts,
                "X_val":X_val,
                "X_val_counts":X_val_counts,
                "X_gen":X_gen,
                "A_val":A_val,
                "nb_samples":nb_samples,
                "sample_temp":sample_temp,
                "alpha_kl":alpha_kl,
                "n_epochs":n_epochs,
                "optim":"adam",
                "learning_rate":learning_rate,
                "weight_decay":weight_decay,
                "keep_fit_history":keep_fit_history,
                "keep_val_history":keep_val_history,
                "keep_gen_vars":True,
                "verbose":not sim_data
                }

        parallel = Parallel(n_jobs=nb_replicates, 
                prefer="processes", verbose=verbose)

        rep_results = parallel(delayed(eval_evomodel)(EvoModelClass,
            model_args, fit_args) for s in range(nb_replicates))

        #
        result_data = dict(
                rep_results=rep_results, # rep for replicates
                )

        if sim_data:
            result_data["logl_data"] = x_logl_data
            if validation:
                result_data["logl_val_data"] = v_logl_data

        dump(result_data, results_file, compress=compress_files)

        # Writing a config file and package versions
        conf_file = output_path+"/{}_conf.ini".format(job_name)
        if not os.path.isfile(conf_file):
            write_conf_packages(config, conf_file)

    ## Report and plot results
    ## #######################
    scores = [result["fit_hist"] for result in rep_results]

    # get min number of epoch of all reps 
    # (maybe some reps stopped before max_iter)
    # to slice the list of epochs with the same length 
    # and be able to cast the list in ndarray        
    min_iter = scores[0].shape[1]
    for score in scores:
        if min_iter >= score.shape[1]: min_iter = score.shape[1]
    the_scores = []
    for score in scores:
        the_scores.append(score[:,:min_iter])

    the_scores = np.array(the_scores)
    #print("The scores {}".format(the_scores.shape))

    ## Generate report file from sampling step
    ## #######################################
    if verbose: print("\nGenerate reports...")

    estim_gens = aggregate_sampled_estimates(
            rep_results, "gen_results")

    report_sampled_estimates(
            estim_gens,
            output_path+"/{}_estim_report".format(job_name),
            )

    ## Ploting results
    ## ###############
    if verbose: print("\nPlotting...")
    plt_elbo_ll_kl_rep_figure(
            the_scores,
            output_path+"/{}_rep_fig".format(job_name),
            sizefont=size_font,
            usetex=plt_usetex,
            print_xtick_every=print_xtick_every,
            title=None,
            plot_validation=validation)

    if validation:
        hist = "val"
    else:
        hist = "fit"

    estimates = aggregate_estimate_values(rep_results,
            "{}_hist_estim".format(hist))
    #return a dictionary of arrays

    # Distance between estimated paramerters 
    # and values given in the config file
    plot_fit_estim_dist(
            estimates, 
            sim_params,
            output_path+"/{}_{}_estim_dist".format(job_name, hist),
            sizefont=size_font,
            usetex=plt_usetex,
            print_xtick_every=print_xtick_every,
            y_limits=[-0.1, 1.1],
            legend='upper right')

    # Correlation between estimated paramerters 
    # and values given in the config file
    plot_fit_estim_corr(
            estimates, 
            sim_params,
            output_path+"/{}_{}_estim_corr".format(job_name, hist),
            sizefont=size_font,
            usetex=plt_usetex,
            print_xtick_every=print_xtick_every,
            y_limits=[-1.1, 1.1],
            legend='lower right')

    if validation:
        # Euclidean and Hamming distances
        euc_keys = ["x_euclidean"]
        ham_keys = ["x_hamming"]
        
        if "a_euclidean" in estimates:
            euc_keys.append("a_euclidean")

        if "a_hamming" in estimates:
            ham_keys.append("a_hamming")

        # Euclidean distance between actual A and inferred A
        # Euclidean distance between actual X and generated X
        plot_fit_seq_dist(
                estimates,
                euc_keys,
                output_path+"/{}_{}_axseq_euclidean_dist".format(
                        job_name, hist),
                sizefont=size_font,
                usetex=plt_usetex,
                print_xtick_every=print_xtick_every,
                y_limits=[-0.1, 1.1],
                y_label="Euclidean distance",
                legend='upper right')

        # Hamming distance between actual A and inferred A
        # Hamming distance between actual X and generated X
        plot_fit_seq_dist(
                estimates,
                ham_keys,
                output_path+"/{}_{}_axseq_hamming_dist".format(
                        job_name, hist),
                sizefont=size_font,
                usetex=plt_usetex,
                print_xtick_every=print_xtick_every,
                y_limits=[-0.1, 1.1],
                y_label="Hamming distance",
                legend='upper right')

    print("\nFin normale du programme\n")
