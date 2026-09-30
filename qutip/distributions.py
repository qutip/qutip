"""
This module provides classes and functions for working with spatial
distributions, such as Wigner distributions, etc.

.. note::

    Experimental.

"""

__all__ = ['Distribution', 'HarmonicOscillatorWaveFunction',
           'HarmonicOscillatorProbabilityFunction']

import numpy as np
from numpy.typing import ArrayLike
from . import isket, ket2dm, Qobj
from .wigner import wigner, qfunc
from ._distributions import psi_fock_multiple_position_complex

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from mpl_toolkits.mplot3d import Axes3D

except ImportError:
    # Type when matplotlib is not installed
    from typing import Any
    Figure = Any
    Axes = Any
    Colormap = Any


class Distribution:
    """A class for representation spatial distribution functions.

    The Distribution class can be used to prepresent spatial distribution
    functions of arbitray dimension (although only 1D and 2D distributions
    are used so far).

    It is indented as a base class for specific distribution function, and
    provide implementation of basic functions that are shared among all
    Distribution functions, such as visualization, calculating marginal
    distributions, etc.

    Parameters
    ----------
    data : array_like
        Data for the distribution. The dimensions must match the lengths of
        the coordinate arrays in xvecs.
    xvecs : list
        List of arrays that spans the space for each coordinate.
    xlabels : list
        List of labels for each coordinate.

    """

    def __init__(self, data: ArrayLike = None, xvecs: list = [],
                 xlabels: list = []):
        self.data = data
        self.xvecs = xvecs
        self.xlabels = xlabels

    def visualize(self, fig: Figure = None, ax: Axes = None,
                  figsize: tuple = (8, 6), colorbar: bool = True,
                  cmap: Colormap = None, style: str = "colormap",
                  show_xlabel: bool = True, show_ylabel: bool = True) -> tuple[Figure, Axes]:
        """
        Visualize the data of the distribution in 1D or 2D, depending
        on the dimensionality of the underlaying distribution.

        Parameters:

        fig : matplotlib Figure instance
            If given, use this figure instance for the visualization,

        ax : matplotlib Axes instance
            If given, render the visualization using this axis instance.

        figsize : tuple
            Size of the new Figure instance, if one needs to be created.

        colorbar: Bool
            Whether or not the colorbar (in 2D visualization) should be used.

        cmap: matplotlib colormap instance
            If given, use this colormap for 2D visualizations.

        style : string
            Type of visualization: 'colormap' (default) or 'surface'.

        show_xlabel : bool
            Whether or not the xlabel is shown.

        show_ylabel : bool
            Whether or not the ylabel is shown.

        Returns
        -------

        fig, ax : tuple
            A tuple of matplotlib figure and axes instances.

        """
        n = len(self.xvecs)
        if n == 2:
            if style == "colormap":
                return self._visualize_2d_colormap(fig=fig, ax=ax,
                                                   figsize=figsize,
                                                   colorbar=colorbar,
                                                   cmap=cmap,
                                                   show_xlabel=show_xlabel,
                                                   show_ylabel=show_ylabel)
            else:
                return self._visualize_2d_surface(fig=fig, ax=ax,
                                                  figsize=figsize,
                                                  colorbar=colorbar,
                                                  cmap=cmap,
                                                  show_xlabel=show_xlabel,
                                                  show_ylabel=show_ylabel)

        elif n == 1:
            return self._visualize_1d(fig=fig, ax=ax, figsize=figsize,
                                      show_xlabel=show_xlabel,
                                      show_ylabel=show_ylabel)
        else:
            raise NotImplementedError(
                f"Distribution visualization in {n} dimensions is not implemented."
            )

    def _visualize_2d_colormap(self, fig=None, ax=None, figsize=(8, 6),
                               colorbar=True, cmap=None,
                               show_xlabel=True, show_ylabel=True):

        if not fig and not ax:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        if cmap is None:
            cmap = mpl.colormaps['RdBu']

        lim = abs(self.data.real).max()

        cf = ax.contourf(self.xvecs[0], self.xvecs[1], self.data.real, 100,
                         norm=mpl.colors.Normalize(-lim, lim),
                         cmap=cmap)

        if show_xlabel:
            ax.set_xlabel(self.xlabels[0], fontsize=12)
        if show_ylabel:
            ax.set_ylabel(self.xlabels[1], fontsize=12)

        if colorbar:
            cb = fig.colorbar(cf, ax=ax)

        return fig, ax

    def _visualize_2d_surface(self, fig=None, ax=None, figsize=(8, 6),
                              colorbar=True, cmap=None,
                              show_xlabel=True, show_ylabel=True):

        if not fig and not ax:
            fig = plt.figure(figsize=figsize)
            ax = Axes3D(fig, azim=-62, elev=25)

        if cmap is None:
            cmap = mpl.colormaps['RdBu']

        lim = abs(self.data.real).max()

        X, Y = np.meshgrid(self.xvecs[0], self.xvecs[1])
        s = ax.plot_surface(X, Y, self.data.real,
                            norm=mpl.colors.Normalize(-lim, lim),
                            rstride=5, cstride=5, cmap=cmap, lw=0.1)

        if show_xlabel:
            ax.set_xlabel(self.xlabels[0], fontsize=12)
        if show_ylabel:
            ax.set_ylabel(self.xlabels[1], fontsize=12)

        if colorbar:
            cb = fig.colorbar(s, ax=ax, shrink=0.5)

        return fig, ax

    def _visualize_1d(self, fig=None, ax=None, figsize=(8, 6),
                      show_xlabel=True, show_ylabel=True):

        if not fig and not ax:
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        p = ax.plot(self.xvecs[0], self.data.real)

        if show_xlabel:
            ax.set_xlabel(self.xlabels[0], fontsize=12)
        if show_ylabel:
            ax.set_ylabel("Marginal distribution", fontsize=12)

        return fig, ax

    def marginal(self, dim=0):
        """
        Calculate the marginal distribution function along the dimension
        `dim`. Return a new Distribution instance describing this reduced-
        dimensionality distribution.

        Parameters
        ----------
        dim : int
            The dimension (coordinate index) along which to obtain the
            marginal distribution.

        Returns
        -------

        d : Distributions
            A new instances of Distribution that describes the marginal
            distribution.

        """
        return Distribution(data=self.data.mean(axis=dim),
                            xvecs=[self.xvecs[dim]],
                            xlabels=[self.xlabels[dim]])

    def project(self, dim=0):
        """
        Calculate the projection (max value) distribution function along the
        dimension `dim`. Return a new Distribution instance describing this
        reduced-dimensionality distribution.

        Parameters
        ----------
        dim : int
            The dimension (coordinate index) along which to obtain the
            projected distribution.

        Returns
        -------
        d : Distributions
            A new instances of Distribution that describes the projection.

        """
        return Distribution(data=self.data.max(axis=dim),
                            xvecs=[self.xvecs[dim]],
                            xlabels=[self.xlabels[dim]])


def _quadrature_functions(x, N, theta):
    """Rows exp(-1j * theta * n) * psi_n(x) for n < N, shape (N, len(x))."""
    return np.exp(-1j * theta * np.arange(N))[:, None] * \
        psi_fock_multiple_position_complex(N - 1, x.astype(complex))


class TwoModeQuadratureCorrelation(Distribution):
    """A class for representing the probability distribution for
    quadrature measurement outcomes given a two-mode wavefunction
    or density matrix.

    Parameters
    ----------
    state : Qobj, default : None
        A quantum state (wavefunction or density matrix) from which the
        distribution is generated.
    theta1 : float, default : 0.0
        Angle for the first coordinate.
    theta2 : float, default : 0.0
        Angle for the second coordinate.
    extent : ArrayLike, default : [[-5, 5], [-5, 5]]
        List of arrays with the bounds [a, b] for each coordinate.
    steps : int, default : 250
        The number of data points generated between the bounds for
        each coordinate.

    """

    def __init__(self, state: Qobj = None, theta1: float = 0.0, theta2: float = 0.0,
                 extent: ArrayLike = [[-5, 5], [-5, 5]], steps: int = 250):

        self.xvecs = [np.linspace(extent[0][0], extent[0][1], steps),
                      np.linspace(extent[1][0], extent[1][1], steps)]

        self.xlabels = [r'$X_1(\theta_1)$', r'$X_2(\theta_2)$']

        self.theta1 = theta1
        self.theta2 = theta2

        if state:
            self.update(state)

    def update(self, state: Qobj):
        """Calculates the probability distribution for quadrature measurement
        outcomes.

        Parameters
        ----------
        state : Qobj
            A quantum state (wavefunction or density matrix) from which the
            distribution is generated.

        """

        if isket(state):
            self.update_psi(state)
        else:
            self.update_rho(state)

    def update_psi(self, psi: Qobj):
        """Calculates the probability distribution for quadrature measurement
        outcomes given a two-mode wavefunction.

        Parameters
        ----------
        psi : Qobj
            A wavefunction from which the distribution is generated.

        """

        N = psi.dims[0][0]
        a1 = _quadrature_functions(self.xvecs[0], N, self.theta1)
        a2 = _quadrature_functions(self.xvecs[1], N, self.theta2)
        self.data = abs(a2.T @ psi.full().reshape(N, N).T @ a1) ** 2

    def update_rho(self, rho: Qobj):
        """Calculates the probability distribution for quadrature measurement
        outcomes given a two-mode density matrix.

        Parameters
        ----------
        rho : Qobj
            A density matrix from which the distribution is generated.

        """

        N = rho.dims[0][0]
        a1 = _quadrature_functions(self.xvecs[0], N, self.theta1)
        a2 = _quadrature_functions(self.xvecs[1], N, self.theta2)
        # rho[(n1, n2), (p1, p2)] contracted with a1[n1] a1*[p1] a2[n2] a2*[p2]
        self.data = np.einsum(
            "abcd,aj,cj,bi,di->ij", rho.full().reshape(N, N, N, N),
            a1, a1.conj(), a2, a2.conj(), optimize=True
        )


class HarmonicOscillatorWaveFunction(Distribution):
    """Calculates and represents the wave function of
       a quantum harmonic oscillator.

    The `HarmonicOscillatorWaveFunction` class computes
    the spatial distribution of the wave function for a quantum
    harmonic oscillator given a set of state coefficients (`psi`).

    By extending the `Distribution` base class, this class
    provides specialized attributes and methods tailored for modeling
    the harmonic oscillator's wave function.This implementation leverages
    the Cython function `psi_fock_multiple_position_complex` from the
    `_distributions.pyx` module to efficiently compute the wave function's
    contribution for each Fock state across spatial coordinates using an
    optimized recurrence relation.

    Parameters
    ----------
    psi : array_like, optional
        Coefficients for each harmonic oscillator state (Fock state) to
        calculate the wave function. Defaults to None, in which case the
        wave function is not initialized until `update` is called.
    omega : float, optional
        The angular frequency of the harmonic oscillator. Defaults to 1.0.
    extent : list, optional
        A list with two elements that defines the range of the spatial
        dimension for calculating the wave function. Defaults to [-5, 5].
    steps : int, optional
        Number of points used to discretize the spatial range defined by
        `extent`. Higher values increase resolution but may slow down
        computations. Defaults to 250.

    Attributes
    ----------
    xvecs : list of arrays
        A list containing arrays that represent the spatial
        coordinates over which the wave function is calculated.
    xlabels : list of str
        A list of labels for each spatial coordinate, in this case with
        one element representing the x-axis.
    omega : float
        The angular frequency of the harmonic oscillator, stored as an
        attribute for use in wave function calculations.
    data : np.ndarray of complex numbers
        The calculated wave function values across the spatial range.
        Populated when `update` is called.

    Methods
    -------
    update(psi)
        Calculates and updates the wave function values for the harmonic
        oscillator based on the provided state coefficients, `psi`.

    References
    ----------
    - Pérez-Jordá, J. M. (2017). On the recursive solution of the quantum
      harmonic oscillator. *European Journal of Physics*, 39(1),
      015402. doi:10.1088/1361-6404/aa9584
    - *Fast-Wave*: High-performance wave function calculations for quantum
       harmonic oscillators.Available at:
       https://github.com/fobos123deimos/fast-wave
    """

    def __init__(self, psi: ArrayLike = None, omega: float = 1.0,
                 extent: list = [-5, 5], steps: int = 250):

        self.xvecs = [np.linspace(extent[0], extent[1], steps)]
        self.xlabels = [r'$x$']
        self.omega = omega

        if psi:
            self.update(psi)

    def update(self, psi: Qobj):
        """Calculate the wavefunction for the given state of an harmonic
        oscillator.

        Parameters
        ----------
        psi : Qobj
            A quantum state from which the distribution is generated.

        """
        rows = psi_fock_multiple_position_complex(
            psi.shape[0] - 1, self.xvecs[0].astype(complex)
        )
        self.data = psi[:, 0] @ rows * pow(self.omega, 0.25)


class HarmonicOscillatorProbabilityFunction(Distribution):
    """A class for representing the probability distribution of a quantum
       harmonic oscillator given a density matrix.

    Parameters
    ----------
    rho : qobj, default : None
        Density matrix for composite quantum systems.
    omega : float, default : 1.0
        The angular frequency of the harmonic oscillator.
    extent : list, default : [[-5, 5], [-5, 5]]
        List of arrays with the bounds [a, b] for each coordinate.
    steps : int, default : 250
        The number of data points generated between the bounds for
        each coordinate.
    """

    def __init__(self, rho: Qobj = None, omega: float = 1.0,
                 extent: list = [-5, 5], steps: int = 250):

        self.xvecs = [np.linspace(extent[0], extent[1], steps)]
        self.xlabels = [r'$x$']
        self.omega = omega

        if rho:
            self.update(rho)

    def update(self, rho: Qobj):
        """Calculates the probability function for the given state of an
        harmonic oscillator (as density matrix).

        Parameters
        ----------
        rho : Qobj
            A density matrix from which the distribution is generated.

        """

        if isket(rho):
            rho = ket2dm(rho)

        rows = psi_fock_multiple_position_complex(
            rho.shape[0] - 1, self.xvecs[0].astype(complex)
        )
        self.data = np.einsum(
            "mx,mn,nx->x", rows, rho.full(), rows.conj()
        ) * pow(self.omega, 0.5)
