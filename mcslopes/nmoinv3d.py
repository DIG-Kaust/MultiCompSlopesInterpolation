from math import floor

import numpy as np
from numba import jit, prange

from pylops import LinearOperator
from pylops.utils.decorators import reshaped


@jit(nopython=True, fastmath=True, nogil=True, parallel=True)
def nmo_forward(data, taxis, yaxis, xaxis, vels_rms):
    dt = taxis[1] - taxis[0]
    ot = taxis[0]
    nt = len(taxis)
    nx = len(xaxis)
    ny = len(yaxis)

    dnmo = np.zeros_like(data)

    # Parallel outer loop on slow axis
    for iy in prange(ny):
        y = yaxis[iy]
        for ix in range(nx):
            x = xaxis[ix]
            h = np.sqrt(y**2+x**2)
            for it0, (t0, vrms) in enumerate(zip(taxis, vels_rms)):
                # Compute NMO traveltime
                tx = np.sqrt(t0**2 + (h / vrms) ** 2)
                it_frac = (tx - ot) / dt  # Fractional index
                it_floor = floor(it_frac)
                it_ceil = it_floor + 1
                w = it_frac - it_floor
                if 0 <= it_floor and it_ceil < nt:  # it_floor and it_ceil must be valid
                    # Linear interpolation
                    dnmo[iy, ix, it0] += (1 - w) * data[iy, ix, it_floor] + w * data[iy, ix, it_ceil]
    return dnmo


@jit(nopython=True, fastmath=True, nogil=True, parallel=True)
def nmo_adjoint(dnmo, taxis, yaxis, xaxis, vels_rms):
    dt = taxis[1] - taxis[0]
    ot = taxis[0]
    nt = len(taxis)
    nx = len(xaxis)
    ny = len(yaxis)

    data = np.zeros_like(dnmo)

    # Parallel outer loop on slow axis; use range if Numba is not installed
    for iy in prange(ny):
        y = yaxis[iy]
        for ix in range(nx):
            x = xaxis[ix]
            h = np.sqrt(y**2+x**2)
            for it0, (t0, vrms) in enumerate(zip(taxis, vels_rms)):
                # Compute NMO traveltime
                tx = np.sqrt(t0**2 + (h / vrms) ** 2)
                it_frac = (tx - ot) / dt  # Fractional index
                it_floor = floor(it_frac)
                it_ceil = it_floor + 1
                w = it_frac - it_floor
                if 0 <= it_floor and it_ceil < nt:
                    # Linear interpolation
                    # In the adjoint, we must spread the same it0 to both it_floor and
                    # it_ceil, since in the forward pass, both of these samples were
                    # pushed onto it0
                    data[iy, ix, it_floor] += (1 - w) * dnmo[iy, ix, it0]
                    data[iy, ix, it_ceil] += w * dnmo[iy, ix, it0]
    return data


class NMO(LinearOperator):
    r"""NMO correction

    3D NMO correction operator to be applied to a dataset of size :math:`n_y \times n_x \times n_t`.

    Parameters
    ----------
    taxis : :obj:`np.ndarray`
       Time axis
    yaxis : :obj:`np.ndarray`
       Crossline axis
    xaxis : :obj:`np.ndarray`
       Inline axis
    vels_rms : :obj:`np.ndarray`
       Velocity profile over time to be used for NMO correction
    dtype : :obj:`str`, optional
       Type of elements in input array.

    Attributes
    ----------
    shape : :obj:`tuple`
       Operator shape
    dims : :obj:`tuple`
       Model dimensions
    dimsd : :obj:`tuple`
       Dats dimensions
    explicit : :obj:`bool`
       Operator contains a matrix that can be solved explicitly
       (``True``) or not (``False``)

   """
    def __init__(self, taxis, yaxis, xaxis, vels_rms, dtype=None):
        self.taxis = taxis
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.vels_rms = vels_rms

        dims = (len(yaxis), len(xaxis), len(taxis))
        if dtype is None:
            dtype = np.result_type(taxis.dtype, yaxis.dtype, vels_rms.dtype)
        super().__init__(dims=dims, dimsd=dims, dtype=dtype)

    @reshaped
    def _matvec(self, x):
        return nmo_forward(x, self.taxis, self.yaxis, self.xaxis, self.vels_rms)

    @reshaped
    def _rmatvec(self, y):
        return nmo_adjoint(y, self.taxis, self.yaxis, self.xaxis, self.vels_rms)
