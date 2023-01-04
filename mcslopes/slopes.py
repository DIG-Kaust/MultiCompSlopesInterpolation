import numpy as np
import pylops

from scipy.ndimage import median_filter
from pylops.basicoperators import FirstDerivative


def analytic_local_slope(d, dt, x, v0, kv):
    """Analytical local slopes for hyperbolic events

    Calculates analytical slope for hyperbolic events with
    linearly increase root-mean square velocity: :math:`v_{rms}(t_0) = v_0 + k_v*t_0`

    Parameters
    ----------
    d : :obj:`np.ndarray`
        Data of size :math:`n_x \times n_t`
    dt : :obj:`float`
        Time sampling
    x : :obj:`np.ndarray`
        Offset axis
    v0 : :obj:`float`
        Initial velocity at :math:`t_0 = 0 s`
    kv : :obj:`float`
        Velocity gradient

    Parameters
    ----------
    slope : :obj:`np.ndarray`
        Local slopes of size :math:`n_x \times n_t`

    """

    [nx, nt] = d.shape
    t = np.arange(nt) * dt 
    
    v_rms = v0 + kv * t
    
    slope = np.zeros((nt, nx))

    for it0 in range(int(nt)):
        for ix in range(nx):
            tapp = np.sqrt(t[it0]**2 + (x[ix]/v_rms[it0])**2)
            it = (np.floor(tapp/dt+1)).astype(int)
            
            if it < nt:
                slope[it, ix] = x[ix] / (tapp * (v_rms[it0])**2)
    
    # Put 0s to max value
    slope[slope == 0.] = np.nanmax(slope)
    
    # Put Nans to 0
    slope[:, 0] = 0.

    return slope


def multicomponent_slopes(data, dx, dt, thresh=1e-3, nmedian=5):
    """Local slopes from multi-component data by direct division

    Calculates local slopes from multi-component data by direct division of spatial and
    time derivative of the data

    Parameters
    ----------
    d : :obj:`np.ndarray`
        Data of size :math:`n_x \times n_t`
    dx : :obj:`float`
        Space sampling
    dt : :obj:`float`
        Time sampling
    thresh : :obj:`float`
        Threshold for mask
    nmedian : :obj:`int`
        Size median filter to apply as post-processing

    Parameters
    ----------
    slope_mc : :obj:`np.ndarray`
        Local slopes of size :math:`n_x \times n_t`

    """

    nx, nt = data.shape
    
    # create mask where no seismic event exists (derivatives are undefined)
    mask = np.ones((nx, nt))
    for ix in range(nx):
        thresh_ix = thresh * np.max(np.abs(data[ix]))
        mask[ix, np.abs(data[ix]) < thresh_ix] = 0.

    # compute time and space derivatives
    Fx = FirstDerivative((nx, nt), axis=0, sampling=dx, order=5, edge=True)
    Ft = FirstDerivative((nx, nt), axis=1, sampling=dt, order=5, edge=True)
    data_dx = Fx * data
    data_dt = Ft * data
    
    # compute slopes
    slope_mc = mask * (-data_dx / data_dt)
    slope_mc[np.isinf(slope_mc)] = 0
    slope_mc[np.isnan(slope_mc)] = 0
    slope_mc = median_filter(slope_mc, size=nmedian).T
    
    return slope_mc


def multicomponent_slopes_inverse(data, dx, dt, reg=1e-1, iter_lim=100):
    """Local slopes from multi-component data by smoothed division

    Calculates local slopes from multi-component data by smoothed division of spatial and
    time derivative of the data

    Parameters
    ----------
    d : :obj:`np.ndarray`
        Data of size :math:`n_x \times n_t`
    dx : :obj:`float`
        Space sampling
    dt : :obj:`float`
        Time sampling
    reg : :obj:`float`
        Regularization parameter for Laplacian
    iter_lim : :obj:`int`
        Number of iterations of LSQR

    Parameters
    ----------
    slope_mc : :obj:`np.ndarray`
        Local slopes of size :math:`n_x \times n_t`

    """
    nx, nt = data.shape
    
    # compute time and space derivatives
    Fx = FirstDerivative((nx, nt), axis=0, sampling=dx, order=5, edge=True)
    Ft = FirstDerivative((nx, nt), axis=1, sampling=dt, order=5, edge=True)
    data_dx = Fx * data
    data_dt = Ft * data

    # compute slopes
    Ddt = pylops.Diagonal(data_dt)
    Lop = pylops.Laplacian((nx, nt))

    slope_mc = pylops.optimization.leastsquares.regularized_inversion(
        Ddt, -data_dx.ravel(), [Lop], epsRs=[reg], **dict(iter_lim=iter_lim))[0]
    slope_mc = slope_mc.reshape(nx, nt).T
    
    return slope_mc


