from setuptools import setup, find_packages
from evoVGM import __version__

_version = __version__

INSTALL_REQUIRES = []

with open("requirements.txt", "r") as fh:
    for line in fh:
        INSTALL_REQUIRES.append(line.rstrip())

setup(
    name='evoVGM',
    version=_version,
    description='Evolutionary-based Variational Generative '+\
            'Model using Multiple Sequence Alignments',
    author='Amine Remita',
    packages=find_packages(),
    scripts=[
        "scripts/evovgm.py",
        "scripts/train_evoVGM_GTR.py",
        ],
    install_requires=INSTALL_REQUIRES
)
