__all__ = ['orbital']

import numpy as np
from scipy.special import sph_harm


def orbital(theta, phi, *args):
    r"""Calculates an angular wave function on a sphere.
    ``psi = orbital(theta,phi,ket1,ket2,...)`` calculates
    the angular wave function on a sphere at the mesh of points
    defined by theta and phi which is
    :math:`\sum_{lm} c_{lm} Y_{lm}(theta,phi)` where :math:`C_{lm}` are the
    coefficients specified by the list of kets. Each ket has 2l+1 components
    for some integer l. The first entry of the ket defines the coefficient
    c_{l,-l}, while the last entry of the ket defines the
    coefficient c_{l, l}.

    Parameters
    ----------
    theta : int/float/list/array
        Polar angles in [0, pi]

    phi : int/float/list/array
        Azimuthal angles in [0, 2*pi]

    args : list/array
        ``list`` of ket vectors.

    Returns
    -------
    ``array`` for angular wave function evaluated at all
              possible combinations of theta and phi

    """
    if isinstance(args[0], list):
        # use the list in args[0]
        args = args[0]

    # convert to numpy array
    theta = np.atleast_1d(theta)
    phi = np.atleast_1d(phi)
    # check that arrays are only 1D
    if len(theta.shape) != 1:
        raise ValueError('Polar angles theta must be 1D list')
    if len(phi.shape) != 1:
        raise ValueError('Azimuthal angles phi must be 1D list')

    # make meshgrid
    phi_mesh, theta_mesh = np.meshgrid(phi, theta)
    # setup empty wavefunction
    psi = np.zeros([theta.shape[0], phi.shape[0]], dtype=complex)
    # iterate through provided kets
    for k in range(len(args)):
        ket = args[k]
        if ket.type == 'bra':
            ket = ket.conj()
        elif not ket.type == 'ket':
            raise TypeError('Invalid type for input ket in orbital')
        # Extract l value from the state
        l = (ket.shape[0] - 1) / 2.0
        if l != np.floor(l):
            raise ValueError(
                'Kets must have odd number of components in orbital')
        l = int(l)
        # get factors from ket
        factors = ket.full()
        # iterate through the possible m

        for i in range(len(factors)):
            # set correct m
            m = i - l
            # calculate spherical harmonics
            # note that theta and phi are interchanged in scipy implementation
            res = sph_harm(m, l, phi_mesh, theta_mesh)
            psi += factors[i] * res

    # flatten output if only one row
    if psi.shape[1] == 1:
        psi = psi.flatten()

    return psi
