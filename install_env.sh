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

# install cusignal
conda install -c rapidsai -c nvidia -c conda-forge \
    cusignal cudatoolkit=11.5 -y
unset CONDA_ALWAYS_YES

# check cupy, cusignal and pylops work as expected
echo 'Checking cupy/cusignal/pylops versions and running a command...'
python -c 'import cupy as cp; print(cp.__version__); cp.ones(10000)*10'
python -c 'import cusignal; print(cusignal.__version__);'
python -c 'import numpy as np; import pylops; print(pylops.__version__); pylops.Identity(10) * np.ones(10)'

echo 'Done!'
