from .seq_collections import SeqCollection

from abc import ABC, abstractmethod
import re

import numpy as np

__all__ = [ 'FullNucCatCollection', 'build_cats',
        'build_cats_Xy_data', 'MSANucCatCollection',
        'build_msa_categorical', 'build_msa_Xy_categorical']

__author__ = "ar"


# #####
# Base collections
# ################

class BaseCollection(ABC):

    def __compute_rep_from_collection(self, sequences):
        for i, seq in enumerate(sequences):
            self._compute_rep_of_sequence(seq.seq._data, i)
            self.ids.append(seq.id)

        return self

    def __compute_rep_from_strings(self, sequences):
        for i, seq in enumerate(sequences):
            self._compute_rep_of_sequence(seq, i)
            self.ids.append(i)

        return self
 
    def _compute_representations(self, sequences):
        if isinstance(sequences, SeqCollection):
            self.__compute_rep_from_collection(sequences)

        else:
            self.__compute_rep_from_strings(sequences)

    @abstractmethod
    def _compute_rep_of_sequence(self, seq, ind):
        """
        """

    # TODO Check for list of SeqCollection
    def check_sequences(self, seqs):
        sequences = []
        if isinstance(seqs, str):
            sequences = [seqs]
        elif isinstance(seqs, list):
            for seq in seqs:
                if isinstance(seq, str):
                    sequences.append(seq)
                else: print("Input object {} is not a string".format(seq))
        elif not isinstance(seqs, SeqCollection):
            raise ValueError("Input sequences should be string, list of string or SeqCollection")

        else : sequences = seqs

        return sequences


class FullNucCatCollection(BaseCollection):

    nuc2cat = {
            'A':[1.,0.,0.,0.], 'G':[0.,1.,0.,0.], 
            'C':[0.,0.,1.,0.], 'T':[0.,0.,0.,1.],
            'U':[0.,0.,0.,1.], 'R':[.5,.5,0.,0.],
            'Y':[0.,0.,.5,.5], 'S':[0.,.5,.5,0.], 
            'W':[.5,0.,0.,.5], 'K':[0.,.5,0.,.5], 
            'M':[.5,0.,.5,0.], 'B':[0.,1/3,1/3,1/3], 
            'D':[1/3,1/3,0.,1/3], 'H':[1/3,0.,1/3,1/3],
            'V':[1/3,1/3,1/3,0.], 'N':[.25,.25,.25,.25]
            }

    def __init__(self, sequences, dtype=np.float32):
        self.dtype = dtype
        self.alphabet = "".join(self.nuc2cat.keys())
        sequences = self.check_sequences(sequences)
        #
        self.ids = []
        self.data = []
        self._compute_representations(sequences)

    def _compute_rep_of_sequence(self, sequence, ind):
        # ind is not used here
        sequence = sequence.upper()
        seq_array = list(re.sub(r'[^'+self.alphabet+']', 'N',
            sequence, flags=re.IGNORECASE))
        seq_cat = np.array([self.nuc2cat[i] for i in seq_array],
                dtype=self.dtype)
        self.data.append(seq_cat)

        return self

class MSANucCatCollection(BaseCollection):
 
    nuc2cat = {
            'A':[1.,0.,0.,0.], 'G':[0.,1.,0.,0.], 
            'C':[0.,0.,1.,0.], 'T':[0.,0.,0.,1.],
            'U':[0.,0.,0.,1.], 'R':[.5,.5,0.,0.],
            'Y':[0.,0.,.5,.5], 'S':[0.,.5,.5,0.], 
            'W':[.5,0.,0.,.5], 'K':[0.,.5,0.,.5], 
            'M':[.5,0.,.5,0.], 'B':[0.,1/3,1/3,1/3], 
            'D':[1/3,1/3,0.,1/3], 'H':[1/3,0.,1/3,1/3],
            'V':[1/3,1/3,1/3,0.], 'N':[.25,.25,.25,.25]
            }

    def __init__(self, sequences, dtype=np.float32):
        self.dtype = dtype
        self.alphabet = "".join(self.nuc2cat.keys())
        sequences = self.check_sequences(sequences)
        #
        self.msa_len = len(sequences[0])
        self.nbseqs = len(sequences)
        self.ids = []
        self.data = np.zeros((self.msa_len, self.nbseqs,4 ), dtype=self.dtype)
        self._compute_representations(sequences)

    def _compute_rep_of_sequence(self, sequence, ind):
        sequence = sequence.upper()
        seq_array = list(re.sub(r'[^'+self.alphabet+']', 'N',
            sequence, flags=re.IGNORECASE))

        assert len(seq_array) == self.msa_len

        for i, ind_char in enumerate(seq_array):
            self.data[i, ind] = self.nuc2cat[ind_char]

        return self

# #####
# Data build functions
# ####################

def build_cats(seq_data, dtype=np.float32):

        return FullNucCatCollection(seq_data, dtype=dtype)

def build_cats_Xy_data(seq_data, dtype=np.float32):
    
    obj_cats = build_cats(seq_data, dtype=dtype)
    X_data = obj_cats.data
    y_data = np.asarray(obj_cats.ids)

    return X_data, y_data

def build_msa_categorical(seq_data, dtype=np.float32):
    return MSANucCatCollection(seq_data, dtype=dtype)

def build_msa_Xy_categorical(seq_data, dtype=np.float32):

    obj_cats = build_msa_categorical(seq_data, dtype=dtype)
    X_data = obj_cats.data
    y_data = np.asarray(obj_cats.ids)

    return X_data, y_data
