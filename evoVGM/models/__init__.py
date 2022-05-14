#################################
##                             ##
##         evoVGM              ##
##      2021 (C) A. R.         ##
##                             ##
#################################

from .evoEncoders.ancestorEncoders import *
from .evoEncoders.branchEncoders import *
from .evoEncoders.kappaEncoders import *
from .evoEncoders.gtrEncoders import *
from .evoDecoders.xDecoders import *
from .evoABCmodels import *
from .evoVGM_JC69 import *
from .evoVGM_K80 import *
from .evoVGM_GTR import *

__author__ = "amine remita"

__all__ = [
        "AncestorDeepDirEncoder",
        "AncestorIndDeepDirEncoder",
        "AncestorDeepCatEncoder",
        "BranchIndGammaEncoder",
        "BranchIndDeepGammaEncoder",
        "BranchIndDeepLogNEncoder",
        "KappaIndDeepGammaEncoder",
        "GTRSubRateIndDirEncoder",
        "GTRSubRateIndDeepDirEncoder",
        "GTRfreqIndDeepDirEncoder",
        #
        "XProbDecoder",
        #
        "BaseEvoVGM",
        "EvoVGM_JC69",
        "EvoVGM_K80",
        "EvoVGM_GTR"
        ]
