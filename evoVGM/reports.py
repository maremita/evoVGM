#import os.path

#import numpy as np
#import pandas as pd

import logomaker as lm
import matplotlib.pyplot as plt
import seaborn as sns
#import matplotlib.cm as cm

__author__ = "amine remita"


def plt_elbo_ll_kl_rep_figure(scores, out_file,
        print_xtick_every=10, usetex=False,
        y_limits=[-10, 0], title="evoModel",
        plot_validation=False):
    fig_format= "png"
    fig_dpi = 150

    fig_file = out_file+"."+fig_format

    sizefont = 10

    f, ax = plt.subplots(figsize=(8, 5))
    ax2 = ax.twinx()

    plt.rcParams.update({'font.size':sizefont, 'text.usetex':usetex})
    plt.subplots_adjust(wspace=0.16, hspace=0.1)

    nb_iters = scores.shape[2] 
    x = [j for j in range(1, nb_iters+1)]

    ax.set_rasterization_zorder(0)
 
    elbo_color = "tomato"
    ll_color = "teal"
    kl_color = "black"

    # plot means
    ax.plot(x, scores[:,0,:].mean(0), "-", color=elbo_color, 
            label="Elbo") # ELBO train
    ax2.plot(x, scores[:,2,:].mean(0), "-", color=kl_color,
            label="KL") # KL train
    ax.plot(x, scores[:,1,:].mean(0), "-", color=ll_color,
            label="Ll") # LL train

    # plot stds
    ax.fill_between(x,
            scores[:,0,:].mean(0)-scores[:,0,:].std(0), 
            scores[:,0,:].mean(0)+scores[:,0,:].std(0), 
            color=elbo_color,
            alpha=0.2, zorder=-1, interpolate=True)

    ax.fill_between(x,
            scores[:,1,:].mean(0)-scores[:,1,:].std(0), 
            scores[:,1,:].mean(0)+scores[:,1,:].std(0),
            color=ll_color,
            alpha=0.2, zorder=-2, interpolate=True)

    ax2.fill_between(x,
            scores[:,2,:].mean(0)-scores[:,2,:].std(0), 
            scores[:,2,:].mean(0)+scores[:,2,:].std(0), 
            color=kl_color,
            alpha=0.2, zorder=-3, interpolate=True)

    # plot validation
    if plot_validation:
        ax.plot(x, scores[:,3,:].mean(0), "-.", color=elbo_color,
                label="Elbo_val") # ELBO train
        ax.plot(x, scores[:,4,:].mean(0), "-.", color=ll_color,
                label="Ll_val") # LL train
        ax2.plot(x, scores[:,5,:].mean(0), "-.", color=kl_color,
                label="KL_val") # KL train
        
        ax.fill_between(x,
                scores[:,3,:].mean(0)-scores[:,3,:].std(0), 
                scores[:,3,:].mean(0)+scores[:,3,:].std(0), 
                color=elbo_color,
                alpha=0.1, zorder=-1, interpolate=True)

        ax.fill_between(x,
                scores[:,4,:].mean(0)-scores[:,4,:].std(0), 
                scores[:,4,:].mean(0)+scores[:,4,:].std(0), 
                color=ll_color,
                alpha=0.1, zorder=-2, interpolate=True)

        ax2.fill_between(x,
                scores[:,5,:].mean(0)-scores[:,5,:].std(0), 
                scores[:,5,:].mean(0)+scores[:,5,:].std(0), 
                color= kl_color,
                alpha=0.1, zorder=-3, interpolate=True)
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

