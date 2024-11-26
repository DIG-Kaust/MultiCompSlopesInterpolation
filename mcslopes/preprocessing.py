import numpy as np

from scipy.signal import butter, filtfilt
from pylops.basicoperators import Diagonal, Restriction, FirstDerivative
from pylops.signalprocessing import FFT2D, FFTND
from mcslopes.nmoinv import NMO
from mcslopes.nmoinv3d import NMO as NMO3D
from mcslopes.slopes import analytic_local_slope, analytic_local_slope3d


def butter_lowpass(cutoff, fs, order=5):
    r"""Butterworth low-pass filter

    Design coefficients of butterworth low-pass filter

    Parameters
    ----------
    cutoff : :obj:`float`
        Cut-off frequency
    fs : :obj:`float`
        Sampling frequency
    order : :obj:`int`
        Order of filter

    Returns
    -------
    b : :obj:`np.ndarray`
        Numerator coefficients
    a : :obj:`np.ndarray`
        Denominator coefficients

    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    r"""Apply Butterworth Low-pass filter

    Apply Butterworth low-pass filter over time axis of input data

    Parameters
    ----------
    data : :obj:`np.ndarray`
        Data of size :math:`n_x \times n_t`
    cutoff : :obj:`float`
        Cut-off frequency
    fs : :obj:`float`
        Sampling frequency
    order : :obj:`int`
        Order of filter

    Returns
    -------
    y : :obj:`np.ndarray`
        Filtered data

    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def mask(data, thresh, itoff=20):
    r"""Apply mask

    Create mask trace-wise using threshold to indentify the start of the mask.

    Parameters
    ----------
    data : :obj:`np.ndarray`
        Data of size :math:`n_x \times n_t`
    thresh : :obj:`float`
        Threshold (the mask excludes everything before the first value larger than ``thresh``)
    itoff : :obj:`int`, optional
        Number of samples used to shift the mask upward

    Returns
    -------
    masktx : :obj:`np.ndarray`
        Mask of size :math:`n_x \times n_t`

    """
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
    r"""Subsample data

    Create restriction operator and apply to data

    Parameters
    ----------
    data : :obj:`np.ndarray`
        Data of size :math:`n_x \times n_t`
    nsub : :obj:`int`
        Subsampling factor
    dtype : :obj:`str`, optional
        Dtype of operator

    Returns
    -------
    data_obs : :obj:`np.ndarray`
        Restricted data of size :math:`(n_x // n_{sub}) \times n_t`
    data_mask : :obj:`np.ndarray`
        Masked data of size :math:`n_x \times n_t`
    Rop : :obj:`pylops.LinearOperator`
        Restriction operator

    """
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
    r"""2D Restriction operator

    Create 2D restriction operator

    Parameters
    ----------
    nx : :obj:`int`
        Number of spatial samples
    nsub : :obj:`int`
        Subsampling factor
    nt : :obj:`int`
        Number of time samples
    dtype : :obj:`str`, optional
        Dtype of operator

    Returns
    -------
    Rop : :obj:`pylops.LinearOperator`
        Restriction operator

    """
    # Identify available traces
    traces_index = np.arange(nx)
    traces_index_sub = traces_index[::nsub]

    # Define restriction operator
    Rop = Restriction(dims=(nx, nt), iava=traces_index_sub, axis=0, dtype=dtype)

    return Rop


def restriction3d(ny, nx, nsub, nt, dtype="float64"):
    r"""3D Restriction operator

    Create 3D restriction operator

    Parameters
    ----------
    ny : :obj:`int`
        Number of crossline samples
    nx : :obj:`int`
        Number of inline samples
    nsub : :obj:`int`
        Subsampling factor of crossline axis
    nt : :obj:`int`
        Number of time samples
    dtype : :obj:`str`, optional
        Dtype of operator

    Returns
    -------
    Rop : :obj:`pylops.LinearOperator`
        Restriction operator

    """
    # identify available traces
    traces_index = np.arange(ny)
    traces_index_sub = traces_index[::nsub]

    # Define restriction operator
    Rop = Restriction(dims=(ny, nx, nt), iava=traces_index_sub, axis=0, dtype=dtype)

    return Rop


def gradient_data(data, nfft_x, nfft_t, dx, dt):
    r"""Gradient data of 2D data

    Compute gradient data of 2D data in frequency-wavenumber domain - i.e.
    apply j*k_x for first derivative and -k_x^2 for second derivative.

    Parameters
    ----------
    data : :obj:`np.ndarray`
        Data of size :math:`n_x \times n_t`
    nfft_x : :obj:`int`
        Number of samples in wavenumber axis
    nfft_t : :obj:`int`
        Number of samples in frequency axis
    dx : :obj:`float`
        Spatial sampling
    dt : :obj:`float`
        Time sampling

    Returns
    -------
    d1 : :obj:`np.ndarray`
        First gradient data of size :math:`n_x \times n_t`
    d2 : :obj:`np.ndarray`
        Second gradient data of size :math:`n_x \times n_t`
    sc1 : :obj:`float`
        Scaling for first gradient data
    sc2 : :obj:`np.ndarray`
        Scaling for second gradient data
    Fop : :obj:`pylops.LinearOperator`
        2D Fourier operator
    D1op : :obj:`pylops.LinearOperator`
        First gradient scaling operator in FK domain
    D2op : :obj:`pylops.LinearOperator`
        Second gradient scaling operator in FK domain
    D : :obj:`np.ndarray`
        FK spectrum of data
    D1 : :obj:`np.ndarray`
        FK spectrum of first gradient data
    D2 : :obj:`np.ndarray`
        FK spectrum of second gradient data
    ks : :obj:`np.ndarray`
        Spatial wavenumber axis
    f : :obj:`np.ndarray`
        Frequency axis

    """
    nx, nt = data.shape
    f = np.fft.fftfreq(nfft_t, dt)
    ks = np.fft.fftfreq(nfft_x, dx)
    Fop = FFT2D(dims=(nx, nt), nffts=(nfft_x, nfft_t), dtype=np.complex128)

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
    r"""Gradient of NMO corrected data

    Compute gradient of NMO corrected data using the gradient data directly as in
    Robertsson et al., 2008, On the use of multicomponent streamer recordings for reconstruction of
    pressure wavefields in the crossline direction. Geophysics.

    Parameters
    ----------
    data : :obj:`np.ndarray`
        Data of size :math:`n_x \times n_t`
    grad : :obj:`np.ndarray`
        First gradient data of size :math:`n_x \times n_t`
    t : :obj:`np.ndarray`
        Time axis
    x : :obj:`np.ndarray`
        Spatial axis
    vnmo : :obj:`float`, optional
        NMO velocity

    Returns
    -------
    gradnmo : :obj:`np.ndarray`
        Gradient of NMO corrected data size :math:`n_x \times n_t`
    pxnmo : :obj:`np.ndarray`
        NMO corrected gradient data
    ptnmo : :obj:`float`
        NMO corrected time derivative data
    dt_dx : :obj:`np.ndarray`
        Derivative of NMO equation over space

    """
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
    r"""Gradient data of 3D data

    Compute crossline gradient data of 3D data in frequency-wavenumber domain - i.e.
    apply j*k_y for first derivative and -k_y^2 for second derivative.

    Parameters
    ----------
    data : :obj:`np.ndarray`
        Data of size :math:`n_y \times n_x \times n_t`
    nfft_y : :obj:`int`
        Number of samples in wavenumber crossline axis
    nfft_x : :obj:`int`
        Number of samples in wavenumber inline axis
    nfft_t : :obj:`int`
        Number of samples in frequency axis
    dy : :obj:`float`
        Crossline sampling
    dx : :obj:`float`
        Inline sampling
    dt : :obj:`float`
        Time sampling
    dtype : :obj:`str`, optional
        Dtype of operators
    computegraddata : :obj:`bool`, optional
        Compute gradients (``True``) or just create operators (``False``)

    Returns
    -------
    d1 : :obj:`np.ndarray`
        First gradient data of size :math:`n_x \times n_t`
    d2 : :obj:`np.ndarray`
        Second gradient data of size :math:`n_x \times n_t`
    sc1 : :obj:`float`
        Scaling for first gradient data
    sc2 : :obj:`np.ndarray`
        Scaling for second gradient data
    Fop : :obj:`pylops.LinearOperator`
        2D Fourier operator
    D1op : :obj:`pylops.LinearOperator`
        First gradient scaling operator in FK domain
    D2op : :obj:`pylops.LinearOperator`
        Second gradient scaling operator in FK domain
    D : :obj:`np.ndarray`
        FK spectrum of data
    D1 : :obj:`np.ndarray`
        FK spectrum of first gradient data
    D2 : :obj:`np.ndarray`
        FK spectrum of second gradient data
    kys : :obj:`np.ndarray`
        Crossline wavenumber axis
    kxs : :obj:`np.ndarray`
        Inline wavenumber axis
    f : :obj:`np.ndarray`
        Frequency axis

    """
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
    r"""Gradient of 3D NMO corrected data

    Compute gradient of 3D NMO corrected data over crossline direction using the gradient data directly as in
    Robertsson et al., 2008, On the use of multicomponent streamer recordings for reconstruction of
    pressure wavefields in the crossline direction. Geophysics.

    Parameters
    ----------
    data : :obj:`np.ndarray`
        Data of size :math:`n_y \times n_x \times n_t`
    grad : :obj:`np.ndarray`
        First gradient data of size :math:`n_y \times n_x \times n_t`
    t : :obj:`np.ndarray`
        Time axis
    y : :obj:`np.ndarray`
        Crossline axis
    x : :obj:`np.ndarray`
        Inline axis
    vnmo : :obj:`float`, optional
        NMO velocity

    Returns
    -------
    gradnmo : :obj:`np.ndarray`
        Gradient of NMO corrected data size :math:`n_y \times n_x \times n_t`
    pynmo : :obj:`np.ndarray`
        NMO corrected gradient data
    ptnmo : :obj:`float`
        NMO corrected time derivative data
    dt_dy : :obj:`np.ndarray`
        Derivative of NMO equation over crossline

    """
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
    r"""FK filter mask

    Design mask to be applied in FK domain to filter energy outside of the chosen signal cone
    based on the following relation ``|k_x| < f / v``.

    Parameters
    ----------
    f : :obj:`np.ndarray`
        Frequency axis
    ks : :obj:`np.ndarray`
        Spatial wavenumber axis
    vel : :obj:`float`
        Maximum velocity to retain
    fmax : :obj:`float`, optional
        Maximum frequency to retain
    critical : :obj:`float`, optional
        Critical angle (used to proportionally adjust velocity and therefore the wavenumber cut-off )
    koffset : :obj:`float`, optional
        Offset to apply over the wavenumber axis to the mask

    Returns
    -------
    mask : :obj:`np.ndarray`
        Mask of size :math:`n_{ks} \times n_f`

    """
    nfft_t = f.size
    fmask = np.zeros(nfft_t)
    fmask[np.abs(f) < fmax] = 1

    [kx, ff] = np.meshgrid(ks, f, indexing='ij')
    mask = np.abs(kx) < (critical * np.abs(ff) / vel + koffset)
    mask = mask.T
    mask *= fmask[:, np.newaxis].astype(bool)
    mask = mask.astype(np.float64)

    return mask


def fk_filter_design3d(f, kys, kxs, vel, fmax, critical=1.00, koffset=0.002):
    r"""3D FK filter mask

    Design mask to be applied in 3D FK domain to filter energy outside of the chosen signal cone
    based on the following relation ``|k_y| < f / v``.

    Parameters
    ----------
    f : :obj:`np.ndarray`
        Frequency axis
    kys : :obj:`np.ndarray`
        Crossline wavenumber axis
    kxs : :obj:`np.ndarray`
        Inline wavenumber axis
    vel : :obj:`float`
        Maximum velocity to retain
    fmax : :obj:`float`, optional
        Maximum frequency to retain
    critical : :obj:`float`, optional
        Critical angle (used to proportionally adjust velocity and therefore the wavenumber cut-off )
    koffset : :obj:`float`, optional
        Offset to apply over the wavenumber axis to the mask

    Returns
    -------
    mask : :obj:`np.ndarray`
        Mask of size :math:`n_{ky} \times n_{kx} \times n_f`

    """
    nfft_t = f.size
    fmask = np.zeros(nfft_t)
    fmask[np.abs(f) < fmax] = 1

    [ky, kx, ff] = np.meshgrid(kys, kxs, f, indexing='ij')
    mask = np.sqrt(np.abs(ky) ** 2 + np.abs(kx) ** 2) < (critical * np.abs(ff) / vel + koffset)
    mask *= fmask[np.newaxis, np.newaxis, :].astype(bool)
    mask = mask.astype(np.float64)

    return mask