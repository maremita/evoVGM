from evoVGM.utils import compute_corr

#import os.path
import numpy as np
#import pandas as pd

import logomaker as lm
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import pdist


__author__ = "amine remita"


def plt_elbo_ll_kl_rep_figure(
        scores,
        out_file,
        print_xtick_every=10,
        usetex=False,
        y_limits=[-10, 0],
        title="evoModel",
        plot_validation=False):

    fig_format= "png"
    fig_dpi = 300

    fig_file = out_file+"."+fig_format

    sizefont = 10

    f, ax = plt.subplots(figsize=(8, 5))
    ax2 = ax.twinx()

    plt.rcParams.update({'font.size':sizefont, 'text.usetex':usetex})
    plt.subplots_adjust(wspace=0.16, hspace=0.1)

    nb_iters = scores.shape[2] 
    x = [j for j in range(1, nb_iters+1)]

    ax.set_rasterization_zorder(0)
 
    elbo_color = "#E9002D"  #sharop red
    ll_color =   "#226E9C"  # darker blue
    kl_color =   "#7C1D69"  # pink

    elbo_color_v = "tomato"
    ll_color_v =   "#009ADE" # light blue
    kl_color_v =   "#AF58BA"  # light pink
    
    # plot means
    ax.plot(x, scores[:,0,:].mean(0), "-", color=elbo_color, 
            label="ELBO", zorder=6) # ELBO train
    ax.plot(x, scores[:,1,:].mean(0), "-", color=ll_color,
            label="LogL", zorder=4) # LL train
    ax2.plot(x, scores[:,2,:].mean(0), "-", color=kl_color,
            label="KL_qp", zorder=4) # KL train

    # plot stds 
    ax.fill_between(x,
            scores[:,0,:].mean(0)-scores[:,0,:].std(0), 
            scores[:,0,:].mean(0)+scores[:,0,:].std(0), 
            color=elbo_color,
            alpha=0.2, zorder=5, interpolate=True)

    ax.fill_between(x,
            scores[:,1,:].mean(0)-scores[:,1,:].std(0), 
            scores[:,1,:].mean(0)+scores[:,1,:].std(0),
            color=ll_color,
            alpha=0.2, zorder=3, interpolate=True)

    ax2.fill_between(x,
            scores[:,2,:].mean(0)-scores[:,2,:].std(0), 
            scores[:,2,:].mean(0)+scores[:,2,:].std(0), 
            color=kl_color,
            alpha=0.2, zorder=-6, interpolate=True)

    # plot validation
    if plot_validation:
        ax.plot(x, scores[:,3,:].mean(0), "-.", color=elbo_color_v,
                label="ELBO_val", zorder=2) # ELBO val
        ax.plot(x, scores[:,4,:].mean(0), "-.", color=ll_color_v,
                label="LogL_val", zorder=0) # LL val
        ax2.plot(x, scores[:,5,:].mean(0), "-.", color=kl_color_v,
                label="KL_qp_val", zorder=2) # KL val
        
        ax.fill_between(x,
                scores[:,3,:].mean(0)-scores[:,3,:].std(0), 
                scores[:,3,:].mean(0)+scores[:,3,:].std(0), 
                color=elbo_color_v,
                alpha=0.1, zorder=1, interpolate=True)

        ax.fill_between(x,
                scores[:,4,:].mean(0)-scores[:,4,:].std(0), 
                scores[:,4,:].mean(0)+scores[:,4,:].std(0), 
                color=ll_color_v,
                alpha=0.1, zorder=2, interpolate=True)

        ax2.fill_between(x,
                scores[:,5,:].mean(0)-scores[:,5,:].std(0), 
                scores[:,5,:].mean(0)+scores[:,5,:].std(0), 
                color= kl_color_v,
                alpha=0.1, zorder=1, interpolate=True)

    ax.set_zorder(ax2.get_zorder()+1)
    ax.set_frame_on(False)

    #ax.set_ylim(y_limits)
    ax.set_ylim([None, 0])
    ax.set_xticks([t for t in range(1, nb_iters+1) if t==1 or\
            t % print_xtick_every==0])
    ax.set_xlabel("Iterations")
    ax.grid()

    handles,labels = [],[]
    for ax in f.axes:
        for h,l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    plt.legend(handles, labels, bbox_to_anchor=(1.1, 1), 
            loc='upper left', borderaxespad=0.)
    plt.suptitle(title)


    plt.savefig(fig_file, bbox_inches="tight", 
            format=fig_format, dpi=fig_dpi)

    plt.close(f)


def plot_fit_estim_dist(
        scores,
        sim_params,
        out_file,
        print_xtick_every=10,
        usetex=False,
        y_limits=[-10, 0],
        title="",
        ):
    """
    scores here is a dictionary of estimate arrays.
    Each array has the shape : (nb_reps, nb_epochs, *estim_shape)
    """

    fig_format= "png"
    fig_dpi = 300

    fig_file = out_file+"."+fig_format

    sizefont = 10

    f, ax = plt.subplots(figsize=(8, 5))

    plt.rcParams.update({'font.size':sizefont, 'text.usetex':usetex})
    plt.subplots_adjust(wspace=0.16, hspace=0.1)

    nb_iters = scores["b"].shape[1]
    x = [j for j in range(1, nb_iters+1)]

    params = {
            "b":"Branch lengths",
            "r":"Rates", 
            "f":"Frequencies",
            "k":"Kappa"}

    colors = { 
            "b":"#226E9C",
            "r":"#D12959", 
            "f":"#40AD5A",
            "k":"#FFAA00"}

    for ind, name in enumerate(scores):
        if name in params:
            estim_scores = scores[name]
            sim_param = sim_params[name].reshape(1,1,-1)
            #print(name, estim_scores.shape)

            # eucl dist
            dists = np.linalg.norm(
                    sim_param - estim_scores, axis=-1)
            #print(name, dists.shape)

            m = dists.mean(0)
            s = dists.std(0)
            ax.plot(x, m, "-", color=colors[name], label=params[name])

            ax.fill_between(x, m-s, m+s, 
                    color=colors[name],
                    alpha=0.2, interpolate=True)
        
    ax.set_xticks([t for t in range(1, nb_iters+1) if t==1 or\
            t % print_xtick_every==0])
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Euclidean distance")
    ax.grid()

    handles,labels = [],[]
    for ax in f.axes:
        for h,l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    plt.legend(handles, labels, bbox_to_anchor=(1.02, 1), 
            loc='upper left', borderaxespad=0.)
    plt.suptitle(title)

    plt.savefig(fig_file, bbox_inches="tight", 
            format=fig_format, dpi=fig_dpi)

    plt.close(f)


def plot_fit_estim_corr(
        scores,
        sim_params,
        out_file,
        print_xtick_every=10,
        usetex=False,
        y_limits=[-10, 0],
        title="",
        ):
    """
    scores here is a dictionary of estimate arrays.
    Each array has the shape : (nb_reps, nb_epochs, *estim_shape)
    """

    fig_format= "png"
    fig_dpi = 300

    fig_file = out_file+"."+fig_format

    sizefont = 10

    f, ax = plt.subplots(figsize=(8, 5))

    plt.rcParams.update({'font.size':sizefont, 'text.usetex':usetex})
    plt.subplots_adjust(wspace=0.16, hspace=0.1)

    nb_iters = scores["b"].shape[1]
    x = [j for j in range(1, nb_iters+1)]

    params = {
            "b":"Branch lengths",
            "r":"Rates", 
            "f":"Frequencies"}
 
    colors = { 
            "b":"#226E9C",
            "r":"#D12959", 
            "f":"#40AD5A",
            "k":"#FFAA00"}

    # Don't compute correlation if vector has the same values
    skip = []
    for name in sim_params:
        if np.all(sim_params[name]==sim_params[name][0]):
            skip.append(name)

    for ind, name in enumerate(scores):
        if name in params and name not in skip:
            estim_scores = scores[name]
            sim_param = sim_params[name]
            #print(name, estim_scores.shape)

            # pearson correlation coefficient
            corrs = compute_corr(sim_param, estim_scores)
            #print(name, corrs.shape)

            m = corrs.mean(0)
            s = corrs.std(0)
            ax.plot(x, m, "-", color=colors[name], label=params[name])

            ax.fill_between(x, m-s, m+s, 
                    color=colors[name],
                    alpha=0.2, interpolate=True)
    
    ax.set_xticks([t for t in range(1, nb_iters+1) if t==1 or\
            t % print_xtick_every==0])
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Correlation coefficient")
    ax.grid()

    handles,labels = [],[]
    for ax in f.axes:
        for h,l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    plt.legend(handles, labels, bbox_to_anchor=(1.01, 1), 
            loc='upper left', borderaxespad=0.)
    plt.suptitle(title)

    plt.savefig(fig_file, bbox_inches="tight", 
            format=fig_format, dpi=fig_dpi)

    plt.close(f)


def aggregate_estimate_values(
        rep_results,
        key, #val_hist_estim
        ):

    #return a dictionary of arrays
    estimates = dict()

    # List (nb_reps) of list (nb_epochs) of dictionaries (estimates) 
    estim_reps = [result[key] for result in rep_results]

    param_names = ["b", "r", "f", "k"]
    names = param_names+["a", "x"]

    estim_shapes = dict()

    nb_reps = len(rep_results)
    nb_epochs = len(estim_reps[0])

    #print(list(estim_reps[0][0].keys()))

    for name in names:
        if name in estim_reps[0][0]:
            #print(name)
            
            estim = estim_reps[0][0][name]
            if name in param_names:
                shape = list(estim.flatten().shape)
            else:
                shape = list(estim.shape)
            #print(name, shape)

            estim_shapes[name] = shape
            #print(shape)

            estimates[name] = np.zeros((nb_reps, nb_epochs, *shape))
            #print(estimates[name].shape)

    for i, replicat in enumerate(estim_reps): # list of reps
        #print("replicat {}".format(type(replicat)))
        for j, epoch in enumerate(replicat): # list of epochs
            #print("epoch {}".format(type(epoch)))
            for name in names:
                if name in epoch:
                    estimation = epoch[name].cpu().detach().numpy()

                    if name in param_names:
                        #print(name, estimation.shape)
                        estimation = estimation.flatten()

                    estimates[name][i,j] = estimation
                    #print(name, estimation.shape)
                    #print(name, estimates[name].shape)

    return estimates 
