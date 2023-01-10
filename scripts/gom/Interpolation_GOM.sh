#!/bin/sh

# Interpolation of GOM data with 1st order grad
python Interpolation_slopes.py -c ../../config/Interpolation_GOM_exp1.yaml

# Interpolation of GOM data with 1st+2nd order grad
python Interpolation_slopes.py -c ../../configInterpolation_GOM_exp2.yaml