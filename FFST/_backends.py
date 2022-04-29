"""
Short description.

@author: fwrnke
@email:  fwrnke@mailbox.org, fwar378@aucklanduni.ac.nz
@date:

"""
from types import ModuleType
from importlib import util
cupy_enabled = util.find_spec('cupy') is not None
scipy_enabled = util.find_spec('scipy') is not None

import numpy as np
if cupy_enabled:
    import cupy as cp


def get_module(xp):
    """
    Find the array module to use.

    Parameters
    ----------
    xp : ModuleType, str
        DESCRIPTION.

    Returns
    -------
    module
        Either `cupy` or `numpy`.

    """
    if isinstance(xp, ModuleType):
        return xp
    
    if cupy_enabled and xp == 'cupy':
        return cp
    else:
        return np

def get_array_module(x):
    """
    Find the module based on provided array.

    Parameters
    ----------
    x : array
        Input array to determine.

    Returns
    -------
    module
        Either `cupy` or `numpy`.

    """
    if cupy_enabled:
        return cp.get_array_module(x)
    else:
        return np
    
def get_module_name(xp):
    """
    Find the module's name.

    Parameters
    ----------
    xp : ModuleType, array
        Input module or array.

    Returns
    -------
    str
        Name of module.

    """
    if isinstance(xp, ModuleType):
        return xp.__name__
    else:
        return get_array_module(xp).__name__
    