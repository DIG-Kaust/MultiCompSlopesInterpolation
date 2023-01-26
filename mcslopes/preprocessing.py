import numpy as np

from scipy.signal import butter, filtfilt
from pylops.basicoperators import Diagonal, Restriction, FirstDerivative
from pylops.signalprocessing import FFT2D, FFTND
from mcslopes.nmoinv import NMO
from mcslopes.nmoinv3D import NMO as NMO3D
from mcslopes.slopes import analytic_local_slope, analytic_local_slope3d


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def mask(data, thresh, itoff=20):
    nx, nt = data.shape
    masktx = np.ones((nx, nt))
    for ix in range(nx):
        itmask = np.where(np.abs(data[ix]) > thresh)[0]
        if len(itmask) > 0:
            masktx[ix, :max(0, itmask[0] - itoff)] = 0.
        else:
            masktx[ix] = masktx[ix - 1]
    return masktx


def subsample(data, nsub, dtype="float64"):
    # identify available traces
    nx = data.shape[0]
    traces_index = np.arange(nx)
    traces_index_sub = traces_index[::nsub]

    # Define restriction operator
    Rop = Restriction(dims=data.shape, iava=traces_index_sub, axis=0, dtype=dtype)

    # Apply restriction operator
    data_obs = Rop * data
    data_mask = Rop.mask(data.ravel())

    return data_obs, data_mask, Rop


def restriction(nx, nsub, nt, dtype="float64"):
    # identify available traces
    traces_index = np.arange(nx)
    traces_index_sub = traces_index[::nsub]

    # Define restriction operator
    Rop = Restriction(dims=(nx, nt), iava=traces_index_sub, axis=0, dtype=dtype)

    return Rop

def restriction3d(ny, nx, nsub, nt, dtype="float64"):
    # identify available traces
    traces_index = np.arange(ny)
    traces_index_sub = traces_index[::nsub]

    # Define restriction operator
    Rop = Restriction(dims=(ny, nx, nt), iava=traces_index_sub, axis=0, dtype=dtype)

    return Rop


def gradient_data(data, nfft_x, nfft_t, dx, dt):
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


def gradient_nmo_data(data, grad, t, x, vnmo=1500.):

    nx, nt = data.shape
    dt = t[1] - t[0]
    NMOOp = NMO(t, x, vnmo * np.ones(nt))

    # NMO of px
    pxnmo = NMOOp @ grad

    # NMO of pt
    Dtop = FirstDerivative((nx, nt), axis=1, sampling=dt, order=5, edge=True)
    pt = Dtop @ data

    ptnmo = NMOOp @ pt
    dt_dx = analytic_local_slope(data, dt, x, vnmo, 0, False)

    # NMO of grad data
    gradnmo = np.real(pxnmo + ptnmo * dt_dx.T)

    return gradnmo, pxnmo, ptnmo, dt_dx


def gradient_data3d(data, nfft_y, nfft_x, nfft_t, dy, dx, dt, dtype="complex128", computegraddata=True):
    ny, nx, nt = data.shape
    f = np.fft.rfftfreq(nfft_t, dt)
    kys = np.fft.fftfreq(nfft_x, dy)
    kxs = np.fft.fftfreq(nfft_x, dx)
    Fop = FFTND(dims=(ny, nx, nt), nffts=(nfft_y, nfft_x, nfft_t),
                sampling=[dy, dx, dt], real=True, dtype=dtype)

    # Compute derivatives in FK domain
    coeff1 = 1j * 2 * np.pi * kys
    coeff2 = -(2 * np.pi * kys) ** 2

    coeff1 = np.repeat(coeff1[:, np.newaxis], nfft_x, axis=1)
    coeff1 = np.repeat(coeff1[:, :, np.newaxis], nfft_t // 2 + 1, axis=2)
    coeff2 = np.repeat(coeff2[:, np.newaxis], nfft_x, axis=1)
    coeff2 = np.repeat(coeff2[:, :, np.newaxis], nfft_t // 2 + 1, axis=2)

    D1op = Diagonal(coeff1.astype(dtype), dtype=dtype)
    D2op = Diagonal(coeff2.astype(dtype), dtype=dtype)

    d1, d2, sc1, sc2, D, D1, D2 = None, None, None, None, None, None, None
    if computegraddata:
        # apply FK transform to data
        D = Fop * data

        D1 = D1op * D
        D2 = D2op * D

        d1 = np.real(Fop.H * D1)
        d2 = np.real(Fop.H * D2)

        # Compute scalars that normalize gradients to data
        sc1 = np.max(np.abs(data)) / np.max(np.abs(d1))
        sc2 = np.max(np.abs(data)) / np.max(np.abs(d2))

    return d1, d2, sc1, sc2, Fop, D1op, D2op, D, D1, D2, kys, kxs, f


def gradient_nmo_data3d(data, grad, t, y, x, vnmo=1500.):

    ny, nx, nt = data.shape
    dt = t[1] - t[0]
    NMOOp = NMO3D(t, y, x, vnmo * np.ones(nt))

    # NMO of py
    pynmo = NMOOp @ grad

    # NMO of pt
    Dtop = FirstDerivative((ny, nx, nt), axis=2, sampling=dt, order=5, edge=True)
    pt = Dtop @ data

    ptnmo = NMOOp @ pt
    dt_dy = analytic_local_slope3d(data, dt, y, x, vnmo, 0, False)

    # NMO of grad data
    gradnmo = np.real(pynmo + ptnmo * dt_dy)

    return gradnmo, pynmo, ptnmo, dt_dy


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


def fk_filter_design3d(f, kys, kxs, vel, fmax, critical=1.00, koffset=0.002):
    nfft_t = f.size
    fmask = np.zeros(nfft_t)
    fmask[np.abs(f) < fmax] = 1

    [ky, kx, ff] = np.meshgrid(kys, kxs, f, indexing='ij')
    mask = np.sqrt(np.abs(ky) ** 2 + np.abs(kx) ** 2) < (critical * np.abs(ff) / vel + koffset)
    mask *= fmask[np.newaxis, np.newaxis, :].astype(bool)
    mask = mask.astype(np.float)

    return mask