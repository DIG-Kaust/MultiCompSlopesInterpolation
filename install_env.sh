#!/bin/bash
# 
# Installer for mcslopes
# 
# Run: ./install.sh
# 
# M. Ravasi, 03/01/2022

echo 'Creating mcslopes environment'

# create conda env
conda env create -f environment.yml
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mcslopes
conda env list
echo 'Created and activated environment:' $(which python)

# install latest version of pylops until v2.4.0 is out!
pip install git+https://github.com/PyLops/pylops.git@dev

# check pylops and cupy work as expected
echo 'Checking pylops/cupy versions and running a command...'
python -c 'import numpy as np; import pylops; print(pylops.__version__); pylops.Identity(10) * np.ones(10)'
python -c 'import cupy as cp; print(cp.__version__); cp.ones(10000)*10' # (Comment out if your system does not have a GPU!!!)

echo 'Done!'
