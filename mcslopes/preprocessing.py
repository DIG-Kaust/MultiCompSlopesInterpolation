import numpy as np
import pylops

from scipy.signal import butter, filtfilt
from pylops.basicoperators import Diagonal, Restriction
from pylops.signalprocessing import FFT2D


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def mask(data, thresh,  itoff=20):
    nx, nt = data.shape

    masktx = np.zeros((nx, nt))
    for ix in range(nx):
        masktx[ix, np.where(data[ix] > thresh * np.max(data[ix]))[0][0] - itoff:] = 1.
    return masktx


def subsample(data, nsub):

    # identify available traces
    nx, nt = data.shape
    traces_index = np.arange(nx)
    traces_index_sub = traces_index[::nsub]

    # Define restriction operator
    Rop = Restriction(dims=(nx, nt), iava=traces_index_sub, axis=0, dtype="float64")

    # Apply restriction operator
    data_obs = Rop * data
    data_mask = Rop.mask(data.ravel())

    return data_obs, data_mask, Rop


def gradient_data(data, nfft_t, nfft_x, dx, dt):
    nx, nt = data.shape
    f = np.fft.fftfreq(nfft_t, dt)
    ks = np.fft.fftfreq(nfft_x, dx)
    Fop = FFT2D(dims=(nx, nt), nffts=(nfft_x, nfft_t), dtype=np.complex)

    # apply FK transform to data
    D = Fop * data

    # Compute derivatives in FK domain
    coeff1 = 1j * 2 * np.pi * ks
    coeff2 = -(2 * np.pi * ks) ** 2
    coeff1 = np.repeat(coeff1[:, np.newaxis], nfft_t, axis=1).ravel()
    coeff2 = np.repeat(coeff2[:, np.newaxis], nfft_t, axis=1).ravel()
    D1op = Diagonal(coeff1)
    D2op = Diagonal(coeff2)

    D1 = (D1op * D.ravel()).reshape(nfft_x, nfft_t)
    D2 = (D2op * D.ravel()).reshape(nfft_x, nfft_t)

    d1 = np.real(Fop.H * D1)
    d2 = np.real(Fop.H * D2)

    # Compute scalars that normalize gradients to data
    sc1 = np.max(np.abs(data)) / np.max(np.abs(d1))
    sc2 = np.max(np.abs(data)) / np.max(np.abs(d2))

    return d1, d2, sc1, sc2, Fop, D1op, D2op, D, D1, D2, ks, f


def fk_filter_design(f, ks, vel, fmax, critical=1.00, koffset=0.002):
    nfft_t = f.size
    fmask = np.zeros(nfft_t)
    fmask[np.abs(f) < fmax] = 1

    [kx, ff] = np.meshgrid(ks, f, indexing='ij')
    mask = np.abs(kx) < (critical * np.abs(ff) / vel + koffset)
    mask = mask.T
    mask *= fmask[:, np.newaxis].astype(bool)
    mask = mask.astype(np.float)

    return mask