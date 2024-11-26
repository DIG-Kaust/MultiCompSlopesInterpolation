#!/usr/bin/env python
# coding: utf-8

import argparse
import yaml

import numpy as np
import matplotlib.pyplot as plt

from pylops.basicoperators import *
from pylops.optimization.sparsity import fista
from pylops.utils.metrics import snr

from mcslopes.nmoinv3d import NMO
from mcslopes.preprocessing import fk_filter_design3d, gradient_data3d, gradient_nmo_data3d, mask, restriction3d
from mcslopes.slopes import multicomponent_slopes_inverse3d
from mcslopes.plotting import plotting_style, explode_volume

plotting_style()


def main(parser):

    ######### INPUT PARAMS #########
    parser.add_argument('-c', '--config', type=str, help='Configuration file')
    args = parser.parse_args()
    with open(args.config, 'r') as stream:
        setup = yaml.load(stream, Loader=yaml.FullLoader)

    USE_CUPY = setup['global']['USE_CUPY']
    if USE_CUPY:
        import cupy as cp
        cp_asarray = cp.asarray
        cp_asnumpy = cp.asnumpy
        np_floatconv = np.float32
        np_floatcconv = np.complex64
    else:
        cp = np
        cp_asarray = np.asarray
        cp_asnumpy = np.asarray
        np_floatconv = np.float64
        np_floatcconv = np.complex128

    filename = setup['exp']['inputfile']
    filename1 = setup['exp']['truefile']
    true_solution = False if filename1 is None else True

    nsub = setup['subsampling']['nsub']
    apply_nmo = setup['preprocessing']['apply_nmo']
    vnmo = setup['preprocessing']['vnmo']
    mask_thresh = setup['preprocessing']['mask_thresh']
    nfft_t = setup['preprocessing']['nfft_t']
    nfft_x = setup['preprocessing']['nfft_x']
    nfft_y = setup['preprocessing']['nfft_y']

    vel = setup['fkmasking']['vel']
    fmax = setup['fkmasking']['fmax']
    critical = setup['fkmasking']['critical']
    koffset = setup['fkmasking']['koffset']

    eps_slopeest = setup['slopeestimation']['eps_slopeest']
    niter_slopeest = setup['slopeestimation']['niter_slopeest']

    eps_slopes = setup['interpolation']['eps_slopes']
    eps_fk = setup['interpolation']['eps_fk']
    niter = setup['interpolation']['niter']

    # Define dictionaries for solvers
    dict_slope_opt = dict(iter_lim=niter_slopeest) if not USE_CUPY else dict(niter=niter_slopeest)

    # Display experiment setup
    sections = ['exp', 'subsampling',
                'preprocessing', 'fkmasking',
                'slopeestimation', 'interpolation']
    print('-------------------------------')
    print('Interpolation with local-slopes')
    print('-------------------------------\n')
    for section in sections:
        print(section.upper())
        for key, value in setup[section].items():
            print(f'{key} = {value}')
        print('\n-------------------------------\n')
    print(f'GPU used: {USE_CUPY}')
    print('-------------------------------\n')


    ######### Data loading and preprocessing #########

    # Load data
    fload = np.load(filename)

    yorig = fload['yorig']
    y = fload['y']
    x = fload['x']
    t = fload['t']
    ys = fload['ys']
    xs = fload['xs']

    dt = t[1] - t[0]
    dy = yorig[1] - yorig[0]
    dx = x[1] - x[0]

    data = fload['data']
    grad = fload['grad1']

    if true_solution:
        fload1 = np.load(filename1)
        dataorig = fload1['data']

    nyorig = yorig.size
    ny, nx, nt = y.size, x.size, t.size
    nfft_t = nt if nfft_t is None else nfft_t
    nfft_x = nx if nfft_x is None else nfft_x
    nfft_y = nx if nfft_y is None else nfft_y

    # Normalize data
    dmax = data.max() * 0.1
    if true_solution:
        dataorig /= dmax
    data /= dmax
    grad /= dmax

    # Create time gain
    gain = (t ** 2)[:, np.newaxis, np.newaxis].T

    # Apply NMO (optional)
    if apply_nmo:
        NMOOporig = NMO(t, yorig - ys, x - xs, vnmo * np.ones(nt))
        NMOOp = NMO(t, y - ys, x - xs, vnmo * np.ones(nt))

        if true_solution: dataorignmo = NMOOporig @ dataorig

        datanmo = NMOOp @ data
        gradnmo = gradient_nmo_data3d(data, grad, t, y-ys, x-xs, vnmo)[0]

        # Overwrite data with nmo corrected data
        data = datanmo.copy()
        grad = gradnmo.copy()
        if true_solution: dataorig = dataorignmo.copy()

    # Create restriction operator
    print(f'Spatial sampling: {dx * nsub}m')
    Rop = restriction3d(nyorig, nx, nsub, nt)
    datagapped = Rop.H @ data

    # Create derivative operators
    _, _, _, _, Fop, D1op, D2op, _, _, _, kys, kxs, f = \
        gradient_data3d(np.zeros((nyorig, nx, nt)), nfft_y, nfft_x, nfft_t, dy, dx, dt, dtype=np_floatcconv, computegraddata=False)

    # Time gain
    data = data * gain
    grad = grad * gain
    datagapped = datagapped * gain
    if true_solution: dataorig = dataorig * gain

    # Compute scalars that normalize gradients to data
    sc1 = np.max(np.abs(data)) / np.max(np.abs(grad))
    print(f'Scalings for 1st derivative data:{sc1:.2f}')

    # Calculate time-space mask
    maskt = mask(datagapped.transpose(1, 0, 2).reshape(nyorig * nx, nt),
                 mask_thresh * np.max(np.abs(datagapped)), 10).reshape(nx, nyorig, nt).transpose(1, 0, 2)

    # FK mask design
    mask_fk = fk_filter_design3d(f, kys, kxs, vel, fmax, critical=critical, koffset=koffset)
    Mf = Diagonal(cp_asarray(mask_fk.ravel().astype(np_floatconv)), dtype=np_floatcconv)

    ######### Multi-channel slope estimation #########

    # Compute slopes
    slope_mc = cp_asnumpy(multicomponent_slopes_inverse3d(Rop.H @ cp_asarray(data), dy, dt,
                                                          graddata=Rop.H @ cp_asarray(grad), Rop=Rop,
                                                          reg=eps_slopeest, **dict_slope_opt)[0])

    ######### Multi-channel interpolation #########

    # Turn to complex since we are now working with a complex model
    Rop.dtype = np.complex128 

    # Let's start by setting the slope regularization term
    D1op0 = FirstDerivative(dims=(nyorig, nx, nt), axis=0, sampling=dy, order=5, edge=True, dtype=np_floatcconv)
    D1op1 = FirstDerivative(dims=(nyorig, nx, nt), axis=2, sampling=dt, order=5, edge=True, dtype=np_floatcconv)
    slope_D1op1 = Diagonal(cp_asarray(slope_mc.ravel()).astype(np_floatconv)) * D1op1
    SRegop = D1op0 + slope_D1op1

    Op = VStack([Rop * Fop.H,
                 sc1 * Rop * Fop.H * D1op,
                 eps_slopes * SRegop * Fop.H]) * Mf
    dtot = cp_asarray(np.concatenate((data.ravel(), sc1 * grad.ravel(),
                                         np.zeros(nyorig * nx * nt)), axis=0).astype(np_floatconv))

    pinv, _, _ = fista(Op, dtot, niter=niter, eps=eps_fk,
                       eigsdict=dict(niter=5, tol=1e-2), show=True)
    dinv = cp_asnumpy(np.real(Fop.H * Mf * pinv)).reshape(nyorig, nx, nt)

    ## Restore data (aka INMO correction)
    if apply_nmo:
        if true_solution: dataorig = NMOOporig.div(dataorig.ravel(), niter=5).reshape(nyorig, nx, nt)
        data = NMOOporig.div(datagapped.ravel(), niter=5).reshape(nyorig, nx, nt)
        dinv = NMOOporig.div(dinv.ravel(), niter=5).reshape(nyorig, nx, nt)
    else:
        data = cp_asnumpy(Rop.H @ cp_asarray(data.ravel())).reshape(nyorig, nx, nt)

    ######### Plotting #########
    snrdata, snrdinv = None, None
    if true_solution:
        snrdata = snr(dataorig, data)
        snrdinv = snr(dataorig, maskt * dinv)

    if true_solution:
        explode_volume(dataorig.transpose(2, 1, 0), x=73, y=52,
                       tlim=[0, t[-1]], tlabel=r'$t$',
                       xlim=[x[0] / 1e3, x[-1] / 1e3], xlabel=r'$x_r$',
                       ylim=[yorig[0] / 1e3, yorig[-1] / 1e3], ylabel=r'$y_r$',
                       labels=('[s]', '[km]', '[km]'),
                       clipval=(-0.5, 0.5), figsize=(8, 8),
                       title=f'Data')

    explode_volume(data.transpose(2, 1, 0), x=73, y=52,
                   tlim=[0, t[-1]], tlabel=r'$t$',
                   xlim=[x[0] / 1e3, x[-1] / 1e3], xlabel=r'$x_r$',
                   ylim=[yorig[0] / 1e3, yorig[-1] / 1e3], ylabel=r'$y_r$',
                   labels=('[s]', '[km]', '[km]'),
                   clipval=(-0.5, 0.5), figsize=(8, 8),
                   title=f'Subsampled Data (SNR={snrdata:.2f} dB)')

    explode_volume(slope_mc.transpose(2, 1, 0), x=73, y=52,
                   tlim=[0, t[-1]], tlabel=r'$t$',
                   xlim=[x[0] / 1e3, x[-1] / 1e3], xlabel=r'$x_r$',
                   ylim=[yorig[0] / 1e3, yorig[-1] / 1e3], ylabel=r'$y_r$',
                   labels=('[s]', '[km]', '[km]'),
                   clipval=(-1 / vel, 1 / vel), figsize=(8, 8), cmap='gist_ncar',
                   title=f'Slopes Sub.')

    explode_volume((maskt * dinv).transpose(2, 1, 0), x=73, y=52,
                   tlim=[0, t[-1]], tlabel=r'$t$',
                   xlim=[x[0] / 1e3, x[-1] / 1e3], xlabel=r'$x_r$',
                   ylim=[yorig[0] / 1e3, yorig[-1] / 1e3], ylabel=r'$y_r$',
                   labels=('[s]', '[km]', '[km]'),
                   clipval=(-0.5, 0.5), figsize=(8, 8),
                   title=f'Reconstructed (SNR={snrdinv:.2f} dB)')

    plt.show()

if __name__ == "__main__":
    description = '3D Multi-Channel Interpolation with local slopes'
    main(argparse.ArgumentParser(description=description))
