from evoVGM.utils import timeSince, dict_to_cpu

from abc import ABC 
import time

import numpy as np
import torch
from scipy.spatial.distance import hamming

__author__ = "amine remita"


class BaseEvoVGM(ABC):

    def generate(self, 
            sites,
            site_counts,
            latent_sample_size=1,
            sample_temp=0.1,
            alpha_kl=0.001,
            keep_vars=False):

        with torch.no_grad():
            if site_counts == None:
                site_counts = torch.ones(sites.shape[0]).to(
                        self.device_)
            # Don't shuffle sites
            return dict_to_cpu(
                    self(
                        sites,
                        site_counts,
                        latent_sample_size=latent_sample_size,
                        sample_temp=sample_temp, 
                        alpha_kl=alpha_kl, 
                        shuffle_sites=False,
                        keep_vars=keep_vars))

    def fit(self,
            X_train,
            X_train_counts,
            latent_sample_size=1,
            sample_temp=0.1,
            alpha_kl=0.001,
            max_iter=100,
            optim="adam",
            optim_learning_rate=0.005, 
            optim_weight_decay=0.1,
            X_val=None,
            # If not None, a validation stpe will be done
            X_val_counts=None,
            A=None, 
            # (nd.array) Embedded ancestral sequence which was used
            # to generate X_val. If not None, it will be used only
            # to report its distance with the inferred ancestral
            # sequence during calidation
            # It is not used during the inference.
            keep_fit_history=False,
            # Save estimates for each epoch of fitting step
            keep_val_history=False,
            # Save estimates for each epoch of validation step
            keep_fit_vars=False,
            # Save estimate variances from fitting step
            keep_val_vars=False,
            # Save estimate variances from validation step
            verbose=None):

        if optim == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), 
                    lr=optim_learning_rate,
                    weight_decay=optim_weight_decay)
        else:
            optimizer = torch.optim.SGD(evoModel.parameters(),
                    lr=optim_learning_rate,
                    weight_decay=optim_weight_decay)

        start = time.time()
        #N_dim = X_train_counts.sum()

        if X_val_counts is not None: N_val_dim = X_val_counts.sum()
        elif X_val is not None: N_val_dim = X_val.shape[0]
 
        self.elbos_list = []
        self.lls_list = []
        self.kls_list = []

        if keep_fit_history: self.fit_estimates = []
        if keep_val_history: self.val_estimates = []

        if X_val is not None:
            np_X_val = X_val.cpu().numpy()
            self.elbos_val_list = []
            self.lls_val_list = []
            self.kls_val_list = []

        for epoch in range(1, max_iter + 1):

            optimizer.zero_grad()
            try:
                fit_dict = self(
                        X_train,
                        X_train_counts,
                        latent_sample_size=latent_sample_size,
                        sample_temp=sample_temp,
                        alpha_kl=alpha_kl,
                        shuffle_sites=True,
                        keep_vars=keep_fit_vars)

                elbos = fit_dict["elbo"]
                lls = fit_dict["logl"].cpu()
                kls = fit_dict["kl_qprior"].cpu()

                loss = - elbos
                loss.backward()
                optimizer.step()
            # Catch some exception (Working on it)
            except Exception as e:
                print("\nStopping training at epoch {} because"\
                        " of an exception in fit()".format(epoch))
                print(e)
                break

            # Validation and printing
            with torch.no_grad():
                if X_val is not None:
                    try:
                        val_dict = self.generate(
                                X_val,
                                X_val_counts, 
                                latent_sample_size=latent_sample_size,
                                sample_temp=sample_temp,
                                alpha_kl=alpha_kl,
                                keep_vars=keep_val_vars)

                        val_dict = dict_to_cpu(val_dict)
                        elbos_val = val_dict["elbo"]
                        lls_val = val_dict["logl"]
                        kls_val = val_dict["kl_qprior"]

                    except Exception as e:
                        print("\nStopping training at epoch {}"\
                                " because of an exception in"\
                                " generate()".format(epoch))
                        print(e)
                        break

                if verbose:
                    if epoch % 10 == 0:
                        chaine = "{}\t Train Epoch: {} \t"\
                                " ELBO: {:.3f}\t Lls {:.3f}\t KLs "\
                                "{:.3f}".format(timeSince(start),
                                        epoch, elbos.item(), 
                                        lls.item(), kls.item())
                        if X_val is not None:
                            chaine += "\nELBO_Val: {:.3f}\t"\
                                    " Lls_Val {:.3f}\t KLs "\
                                    "{:.3f}".format(
                                            elbos_val.item(),
                                            lls_val.item(), 
                                            kls_val.item())
                        print(chaine, end="\r")

                # Add measure values to lists if all is alright
                self.elbos_list.append(elbos.item())
                self.lls_list.append(lls.item())
                self.kls_list.append(kls.item())

                if keep_fit_history:
                    fit_estim = dict()
                    for estim in ["b", "r", "f", "k"]:
                        if estim in fit_dict:
                            fit_estim[estim] = fit_dict[estim].cpu()
                    self.fit_estimates.append(fit_estim)

                if X_val is not None:
                    self.elbos_val_list.append(elbos_val.item())
                    self.lls_val_list.append(lls_val.item())
                    self.kls_val_list.append(kls_val.item())

                    if keep_val_history:
                        val_estim = dict()
                        for estim in ["b", "r", "f", "k"]:
                            if estim in val_dict:
                                val_estim[estim]=val_dict[estim]

                        # Generated sequences and inferred ancestral
                        # sequences are not saved for each epoch
                        # because they consume a lot of space.
                        # Instead of that, we report their distances
                        # with actual sequences.
 
                        # Compute Hamming and average Euclidean distances
                        # between X_val and generated sequences
                        xrecons = val_dict["x"].numpy()
                        x_ham_dist = [hamming(
                            xrecons[:,i,:].argmax(axis=1),
                            np_X_val[:,i,:].argmax(axis=1))\
                                    for i in range(np_X_val.shape[1])]
                        x_euc_dist = np.linalg.norm(xrecons - np_X_val,
                                axis=2).mean(0)
                        val_estim['x_hamming'] = x_ham_dist
                        val_estim['x_euclidean'] = x_euc_dist
 
                        # Compute Hamming and average Euclidean distances
                        # between actual ancestral sequence and inferred
                        # ancestral sequence
                        if A is not None:
                            estim_ancestor = val_dict["a"].numpy()
                            a_ham_dist = hamming(estim_ancestor.argmax(axis=1),
                                    A.argmax(axis=1))
                            a_euc_dist = np.linalg.norm(estim_ancestor - A,
                                    axis=1).mean()
                            val_estim['a_hamming'] = a_ham_dist
                            val_estim['a_euclidean'] = a_euc_dist

                        self.val_estimates.append(val_estim)

        # convert to ndarray to facilitate post-processing
        with torch.no_grad():
            self.elbos_list = np.array(self.elbos_list)
            self.lls_list = np.array(self.lls_list) 
            self.kls_list = np.array(self.kls_list)
            if X_val is not None:
                self.elbos_val_list = np.array(self.elbos_val_list)
                self.lls_val_list = np.array(self.lls_val_list) 
                self.kls_val_list = np.array(self.kls_val_list)
