#################################
##                             ##
##         evoVGM              ##
##      2021 (C) A. R.         ##
##                             ##
#################################

from .evoEncoders.ancestorEncoders import *
from .evoEncoders.branchEncoders import *
from .evoEncoders.gtrEncoders import *
from .evoDecoders.xDecoders import *
from .evoABCmodels import *
#from .evoVGM_JC import *
#from .evoVGM_K80 import *
from .evoVGM_GTR import *

__author__ = "amine remita"

__all__ = [
        "AncestorDeepDirEncoder",
        "AncestorIndDeepDirEncoder",
        "AncestorDeepCatEncoder",
        "BranchIndGammaEncoder",
        "BranchIndDeepGammaEncoder",
        "BranchIndDeepLogNEncoder",
        "GTRSubRateIndDirEncoder",
        "GTRSubRateIndDeepDirEncoder",
        "GTRfreqIndDeepDirEncoder",
        #
        #"XJCProbDecoder",
        #"XK80ProbDecoder",
        "XGTRProbDecoder",
        #
        "BaseEvoVGM",
        #"EvoVGM_JC",
        #"EvoVGM_K80",
        "EvoVGM_GTR"
        ]
