from __future__ import division, print_function, absolute_import

import numpy as np

from .meyerShearlet import (meyerShearletSpect, meyeraux)
from ._scalesShearsAndSpectra import scalesShearsAndSpectra
# from ._fft import fftshift, ifftshift, fftn, ifftn
from ._backends import get_module, get_array_module, get_module_name, cupy_enabled, scipy_enabled

if cupy_enabled:
    import cupy as cp
    import warnings
if scipy_enabled:
    import scipy.fft


def shearletTransformSpect(A, Psi=None, numOfScales=None,
                           realCoefficients=True, maxScale='max',
                           shearletSpect=meyerShearletSpect,
                           shearletArg=meyeraux, realReal=True):
    """Compute the forward shearlet transform.

    If the shearlet spectra, Psi, are not given they are computed using
    parameters guessed from the coefficients.

    Parameters
    ----------
    A : array
        image to transform (2d)
    Psi : array (3d), optional
        spectrum of shearlets (3d)
    realCoefficients : bool, optional
        force real-valued coefficients
    maxScale : {'min', 'max'}
        maximal or minimal finest scale
    shearletSpect : {meyerShearletSpect, meyerSmoothShearletSpect}
        shearlet spectrum to use
    shearletArg : function
        auxiliarry function for the shearlet
    realReal : bool, optional
        return coefficients with real dtype (truncate minimal imaginary
        component).

    Returns
    -------
    ST : array (2d)
        reconstructed image
    Psi : array (3d), optional
        spectrum of shearlets (3d)

    Notes
    -----
    example usage

    # compute shearlet transform of image A with default parameters
    ST, Psi = shearletTransformSpect(A)

    # compute shearlet transform of image A with precomputed shearlet spectrum
    ST, Psi = shearletTransformSpect(A, Psi)

    # compute sharlet transform of image A with specified number of scales
    ST, Psi = shearletTransformSpect(A, numOfScales=4)

    """
    # get array module, i.e cupy or numpy (default)
    xp = get_array_module(A)

    if get_module_name(xp) == 'cupy':
        warnings.warn('Using Cupy!')
        fftn = cp.fft.fftn
        ifftn = cp.fft.ifftn
        fftshift = cp.fft.fftshift
        ifftshift = cp.fft.ifftshift
    elif scipy_enabled and get_module_name(xp) == 'numpy':
        warnings.warn('Using Scipy!')
        fftn = scipy.fft.fftn
        ifftn = scipy.fft.ifftn
        fftshift = scipy.fft.fftshift
        ifftshift = scipy.fft.ifftshift
    else:
        warnings.warn('Using Numpy!')
        fftn = np.fft.fftn
        ifftn = np.fft.ifftn
        fftshift = np.fft.fftshift
        ifftshift = np.fft.ifftshift
    
    # parse input
    A = xp.asarray(A)
    if (A.ndim != 2) or xp.any(xp.asarray(A.shape) <= 1):
        raise ValueError("2D image required")

    # compute spectra
    if Psi is None:
        l = A.shape
        if numOfScales is None:
            numOfScales = int(xp.floor(0.5 * xp.log2(xp.max(l))))
            if numOfScales < 1:
                raise ValueError('image to small!')
        Psi = scalesShearsAndSpectra(l, numOfScales=numOfScales,
                                     realCoefficients=realCoefficients,
                                     shearletSpect=meyerShearletSpect,
                                     shearletArg=meyeraux)
    else:
        Psi = xp.asarray(Psi)

    # shearlet transform
    if False:
        # INCORRECT TO HAVE FFTSHIFT SINCE Psi ISNT SHIFTED!
        uST = Psi * fftshift(fftn(A))[..., xp.newaxis]
        ST = ifftn(ifftshift(uST, axes=(0, 1)), axes=(0, 1))
    else:
        uST = Psi * fftn(A)[..., xp.newaxis]
        ST = ifftn(uST, axes=(0, 1))

    # due to round-off errors the imaginary part is not zero but very small
    # -> neglect it
    if realCoefficients and realReal and xp.isrealobj(A):
        ST = ST.real
    
    if Psi is None:
        return (ST, Psi)
    else:
        return ST
