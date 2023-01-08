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

# install cupy and cusignal (Comment out if your system does not have a GPU!!!)
conda install -c rapidsai -c nvidia -c conda-forge \
    cusignal cudatoolkit=11.5 -y
unset CONDA_ALWAYS_YES

# check pylops, cupy, cusignal work as expected
echo 'Checking pylops/cupy/cusignal versions and running a command...'
python -c 'import numpy as np; import pylops; print(pylops.__version__); pylops.Identity(10) * np.ones(10)'
python -c 'import cupy as cp; print(cp.__version__); cp.ones(10000)*10' # (Comment out if your system does not have a GPU!!!)
python -c 'import cusignal; print(cusignal.__version__);' # (Comment out if your system does not have a GPU!!!)

echo 'Done!'
