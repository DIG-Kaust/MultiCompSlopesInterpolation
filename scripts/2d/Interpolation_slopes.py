#!/usr/bin/env python
# coding: utf-8

import argparse
import yaml

import numpy as np
import matplotlib.pyplot as plt

from pylops.basicoperators import *
from pylops.optimization.sparsity import fista
from pylops.utils.metrics import snr

from mcslopes.nmoinv import NMO
from mcslopes.preprocessing import fk_filter_design, gradient_data, gradient_nmo_data, mask, restriction
from mcslopes.slopes import multicomponent_slopes_inverse


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
    else:
        cp_asarray = np.asarray
        cp_asnumpy = np.asarray

    filename = setup['exp']['inputfile']
    filename1 = setup['exp']['truefile']
    true_solution = False if filename1 is None else True

    nsub = setup['subsampling']['nsub']
    apply_nmo = setup['preprocessing']['apply_nmo']
    vnmo = setup['preprocessing']['vnmo']
    mask_thresh = setup['preprocessing']['mask_thresh']
    nfft_t = setup['preprocessing']['nfft_t']
    nfft_x = setup['preprocessing']['nfft_x']

    vel = setup['fkmasking']['vel']
    fmax = setup['fkmasking']['fmax']
    critical = setup['fkmasking']['critical']
    koffset = setup['fkmasking']['koffset']

    eps_slopeest = setup['slopeestimation']['eps_slopeest']
    niter_slopeest = setup['slopeestimation']['niter_slopeest']

    use_secondder = setup['interpolation']['use_secondder']
    eps_slopes = setup['interpolation']['eps_slopes']
    eps_fk = setup['interpolation']['eps_fk']
    niter = setup['interpolation']['niter']

    # Define dictionaries for solvers
    dict_slope_opt = dict(iter_lim=niter_slopeest) if not USE_CUPY else dict(niter=niter_slopeest)

    # Display experiment setup
    sections = ['exp', 'subsampling',
                'preprocessing', 'fkmasking',
                'slopeestimation', 'interpolation']
    print('----------------------------')
    print('Interpolation with local-slopes')
    print('----------------------------\n')
    for section in sections:
        print(section.upper())
        for key, value in setup[section].items():
            print(f'{key} = {value}')
        print('\n----------------------------\n')
    print(f'GPU used: {USE_CUPY}')
    print('----------------------------\n')


    ######### Data loading and preprocessing #########

    # Load data
    fload = np.load(filename)

    xorig = fload['xorig']
    x = fload['x']
    t = fload['t']
    dt = t[1] - t[0]
    dx = xorig[1] - xorig[0]

    data = fload['data'].T
    grad = fload['grad1'].T
    if use_secondder: grad2 = fload['grad2sub'].T

    if true_solution:
        fload1 = np.load(filename1)
        dataorig = fload1['data'].T

    nxorig = xorig.size
    nx, nt = x.size, t.size

    # Create time gain
    gain = (t / 2.)[:, np.newaxis]

    # Apply NMO (optional)
    if apply_nmo:
        NMOOporig = NMO(t, xorig, vnmo * np.ones(nt))
        NMOOp = NMO(t, x, vnmo * np.ones(nt))

        if true_solution: dataorignmo = NMOOporig @ dataorig

        datanmo = NMOOp @ data
        gradnmo = gradient_nmo_data(data, grad, t, x, vnmo)[0]
        # grad2nmo = # NOT READY YET!

        # Overwrite data with nmo corrected data
        data = datanmo.copy()
        grad = gradnmo.copy()
        #if use_secondder: grad2 = grad2nmo.copy()
        if true_solution: dataorig = dataorignmo.copy()

    # Create restriction operator
    print(f'Spatial sampling: {dx * nsub}m')
    Rop = restriction(nxorig, nsub, nt)

    # Create derivative operators
    _, _, _, _, Fop, D1op, D2op, D, _, _, ks, f = gradient_data(np.zeros((nxorig, nt)), nfft_x, nfft_t, dx, dt)

    # Time gain
    data = data * gain.T
    grad = grad * gain.T
    if true_solution: dataorig = dataorig * gain.T

    # Compute scalars that normalize gradients to data
    sc1 = np.max(np.abs(data)) / np.max(np.abs(grad))
    sc2 = 0.0
    if use_secondder: sc2 = np.max(np.abs(data)) / np.max(np.abs(grad2))
    print(f'Scalings for 1st derivative data:{sc1:.2f}, and 2nd derivative data:{sc2:.2f}')

    # Calculate time-space mask
    maskt = mask(Rop.H @ data, mask_thresh)

    # FK mask design
    mask_fk = fk_filter_design(f, ks, vel, fmax, critical=critical, koffset=koffset)
    Mf = Diagonal(mask_fk.astype(np.complex128).T.ravel(), dtype=np.complex128)

    ######### Multi-channel slope estimation #########
    slope_mc = cp_asnumpy(multicomponent_slopes_inverse(Rop.H @ cp_asarray(data), dx, dt,
                                                        graddata=Rop.H @ cp_asarray(grad), Rop=Rop,
                                                        reg=eps_slopeest,
                                                        **dict_slope_opt))

    ######### Multi-channel interpolation #########

    # Let's start by setting the slope regularization term
    D1op0 = FirstDerivative(dims=(nxorig, nt), axis=0, sampling=dx, order=5, edge=True, dtype="complex128")
    D1op1 = FirstDerivative(dims=(nxorig, nt), axis=1, sampling=dt, order=5, edge=True, dtype="complex128")
    slope_D1op1 = Diagonal(cp_asarray(slope_mc).T.ravel()) * D1op1
    SRegop = D1op0 + slope_D1op1

    if not use_secondder:
        # only first
        F2op = VStack([Rop*Fop.H,
                       sc1*Rop*Fop.H*D1op,
                       eps_slopes * SRegop * Fop.H]) * Mf
        data2 = cp_asarray(np.concatenate((data.ravel(), sc1*grad.ravel(), np.zeros(nt*nxorig)), axis=0))
    else:
        # 1st and 2nd
        F2op = VStack([Rop*Fop.H,
                       sc1*Rop*Fop.H*D1op,
                       sc2*Rop*Fop.H*D2op,
                       eps_slopes * SRegop * Fop.H]) * Mf
        data2 = cp_asarray(np.concatenate((data.ravel(), sc1*grad.ravel(), sc2*grad2.ravel(), np.zeros(nt*nxorig)), axis=0))

    pinv, _, _ = fista(F2op, data2, niter=niter, eps=eps_fk,
                        eigsdict=dict(niter=5, tol=1e-2), show=True)

    dinv = cp_asnumpy(np.real(Fop.H * Mf * pinv).reshape(nxorig, nt))

    ## Restore data (aka INMO correction)
    if apply_nmo:
        if true_solution: dataorig = NMOOporig.div(dataorig.ravel()).reshape(nxorig, nt)
        data = NMOOporig.div(cp_asnumpy(Rop.H @ cp_asarray(data.ravel()))).reshape(nxorig, nt)
        dinv = NMOOporig.div(dinv.ravel()).reshape(nxorig, nt)
    else:
        data = cp_asnumpy(Rop.H @ cp_asarray(data.ravel())).reshape(nxorig, nt)

    ######### Plotting #########

    if true_solution: Dorig = Fop * dataorig
    D = Fop * data
    Drec = Fop * dinv

    vlim = 0.3 * np.abs(data).max()
    Vlim = np.abs(D).max()

    if not true_solution:
        fig, axs = plt.subplots(2, 2, figsize=(12, 12), gridspec_kw={'height_ratios': [2, 1]})
        axs[0, 0].imshow(data.T, cmap='gray', aspect='auto', vmin=-vlim, vmax=vlim,
                         extent=(x[0] / 1000, x[-1] / 1000, t[-1], t[0]))
        axs[0, 0].set_title('Original')
        axs[0, 0].set_xlabel('Offset (m)')
        axs[0, 0].set_ylabel('TWT (s)')
        axs[0, 1].imshow(dinv.T, cmap='gray', aspect='auto', vmin=-vlim, vmax=vlim,
                         extent=(x[0] / 1000, x[-1] / 1000, t[-1], t[0]))
        axs[0, 1].set_title('Reconstructed')
        axs[0, 1].set_xlabel('Offset (m)')

        axs[1, 0].imshow(np.fft.fftshift(np.abs(D).T)[nt // 2:], cmap='gist_ncar_r', aspect='auto', vmin=0, vmax=Vlim,
                         extent=(np.fft.fftshift(ks)[0], np.fft.fftshift(ks)[-1], f[nt // 2 - 1], f[0]))
        axs[1, 0].set_xlim(-1 / (2 * dx), 1 / (2 * dx))
        axs[1, 0].set_ylim(50, 0)
        axs[1, 0].set_xlabel('Wavenumber (1/m)')
        axs[1, 0].set_ylabel('Frequency (Hz)')
        axs[1, 1].imshow(np.fft.fftshift(np.abs(Drec).T)[nt // 2:], cmap='gist_ncar_r', aspect='auto', vmin=0,
                         vmax=Vlim,
                         extent=(np.fft.fftshift(ks)[0], np.fft.fftshift(ks)[-1], f[nt // 2 - 1], f[0]))
        axs[1, 1].set_xlim(-1 / (2 * dx), 1 / (2 * dx))
        axs[1, 1].set_ylim(50, 0)
        axs[1, 1].set_xlabel('Wavenumber (1/m)')

        plt.tight_layout()
    else:
        fig, axs = plt.subplots(2, 3, figsize=(12, 12), gridspec_kw={'height_ratios': [2, 1]})
        axs[0, 0].imshow(dataorig.T, cmap='gray', aspect='auto', vmin=-vlim, vmax=vlim,
                         extent=(x[0]/1000, x[-1]/1000, t[-1], t[0]))
        axs[0, 0].set_title('Original')
        axs[0, 0].set_xlabel('Offset (m)')
        axs[0, 0].set_ylabel('TWT (s)')
        axs[0, 1].imshow(data.T, cmap='gray', aspect='auto', vmin=-vlim, vmax=vlim,
                         extent=(x[0]/1000, x[-1]/1000, t[-1], t[0]))
        axs[0, 1].set_title('Subsampled')
        axs[0, 1].set_xlabel('Offset (m)')
        axs[0, 2].imshow(dinv.T, cmap='gray', aspect='auto', vmin=-vlim, vmax=vlim,
                         extent=(x[0]/1000, x[-1]/1000, t[-1], t[0]))
        axs[0, 2].set_title(f'Reconstructed (SNR={snr(dataorig, maskt * dinv):.2f})')
        axs[0, 2].set_xlabel('Offset (m)')

        axs[1, 0].imshow(np.fft.fftshift(np.abs(Dorig).T)[nt // 2:], cmap='gist_ncar_r', aspect='auto', vmin=0, vmax=Vlim,
                         extent=(np.fft.fftshift(ks)[0], np.fft.fftshift(ks)[-1], f[nt // 2 - 1], f[0]))
        axs[1, 0].set_xlim(-1 / (2 * dx), 1 / (2 * dx))
        axs[1, 0].set_ylim(50, 0)
        axs[1, 0].set_xlabel('Wavenumber (1/m)')
        axs[1, 0].set_ylabel('Frequency (Hz)')
        axs[1, 1].imshow(np.fft.fftshift(np.abs(D).T)[nt // 2:], cmap='gist_ncar_r', aspect='auto', vmin=0, vmax=Vlim,
                         extent=(np.fft.fftshift(ks)[0], np.fft.fftshift(ks)[-1], f[nt // 2 - 1], f[0]))
        axs[1, 1].set_xlim(-1 / (2 * dx), 1 / (2 * dx))
        axs[1, 1].set_ylim(50, 0)
        axs[1, 1].set_xlabel('Wavenumber (1/m)')
        axs[1, 1].set_ylabel('Frequency (Hz)')
        axs[1, 2].imshow(np.fft.fftshift(np.abs(Drec).T)[nt // 2:], cmap='gist_ncar_r', aspect='auto', vmin=0, vmax=Vlim,
                         extent=(np.fft.fftshift(ks)[0], np.fft.fftshift(ks)[-1], f[nt // 2 - 1], f[0]))
        axs[1, 2].set_xlim(-1 / (2 * dx), 1 / (2 * dx))
        axs[1, 2].set_ylim(50, 0)
        axs[1, 2].set_xlabel('Wavenumber (1/m)')

        plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    description = '2D Multi-Channel Interpolation with local slopes'
    main(argparse.ArgumentParser(description=description))
