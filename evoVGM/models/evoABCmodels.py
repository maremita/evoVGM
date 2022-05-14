from evoVGM.utils import timeSince

from abc import ABC 
import time

import numpy as np
import torch

__author__ = "amine remita"


class BaseEvoVGM(ABC):

    def generate(self, 
            sites,
            site_counts,
            latent_sample_size=1,
            sample_temp=0.1,
            alpha_kl=0.001):

        with torch.no_grad():
            if site_counts == None:
                site_counts = torch.ones(sites.shape[0]).to(
                        self.device)
            # Don't shuffle sites
            return self(
                    sites,
                    site_counts,
                    latent_sample_size=latent_sample_size,
                    sample_temp=sample_temp, 
                    alpha_kl=alpha_kl, 
                    shuffle_sites=False)

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
            X_val_counts=None, 
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

        if X_val is not None:
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
                        shuffle_sites=True)

                elbos = fit_dict["elbo"]
                lls = fit_dict["logl"]
                kls = fit_dict["kl_qprior"]

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
                                alpha_kl=alpha_kl)

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
                    if epoch % 10 == 0 or epoch <= 10:
                        chaine = "{}\t Train Epoch: {} \t"\
                                " ELBO: {:.3f}\t Lls {:.3f}\t KLs "\
                                "{:.3f}".format(timeSince(start),
                                        epoch, elbos.item(), 
                                        lls.item(), kls.item())
                        if X_val is not None:
                            chaine += "\n{} \t ELBO_Val: {:.3f}\t"\
                                    " Lls_Val {:.3f}\t KLs "\
                                    "{:.3f}".format(elbos_val.item(),
                                            lls_val.item(), 
                                            kls_val.item())
                        print(chaine, end="\r")

                # Add measure values to lists if all is alright
                self.elbos_list.append(elbos.item())
                self.lls_list.append(lls.item())
                self.kls_list.append(kls.item())
                if X_val is not None:
                    self.elbos_val_list.append(elbos_val.item())
                    self.lls_val_list.append(lls_val.item())
                    self.kls_val_list.append(kls_val.item())

        # convert to ndarray to facilitate post-processing
        with torch.no_grad():
            self.elbos_list = np.array(self.elbos_list)
            self.lls_list = np.array(self.lls_list) 
            self.kls_list = np.array(self.kls_list)
            if X_val is not None:
                self.elbos_val_list = np.array(self.elbos_val_list)
                self.lls_val_list = np.array(self.lls_val_list) 
                self.kls_val_list = np.array(self.kls_val_list)
