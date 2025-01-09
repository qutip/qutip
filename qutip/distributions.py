"""
This module provides classes and functions for working with spatial
distributions, such as Wigner distributions, etc.

.. note::

    Experimental.

"""

__all__ = ['Distribution', 'WignerDistribution', 'QDistribution',
           'TwoModeQuadratureCorrelation',
           'HarmonicOscillatorWaveFunction',
           'HarmonicOscillatorProbabilityFunction']

import numpy as np
from numpy import pi, exp, sqrt

from scipy.special import hermite, factorial

from . import isket, ket2dm, state_number_index
from .wigner import wigner, qfunc
from ._distributions import psi_n_single_fock_multiple_position_complex

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
except:
    pass


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

    def __init__(self, data=None, xvecs=[], xlabels=[]):
        self.data = data
        self.xvecs = xvecs
        self.xlabels = xlabels

    def visualize(self, fig=None, ax=None, figsize=(8, 6),
                  colorbar=True, cmap=None, style="colormap",
                  show_xlabel=True, show_ylabel=True):
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

        Returns
        -------

        fig, ax : tuple
            A tuple of matplotlib figure and axes instances.

        """
        n = len(self.xvecs)
        if n == 2:
            if style == "colormap":
                return self.visualize_2d_colormap(fig=fig, ax=ax,
                                                  figsize=figsize,
                                                  colorbar=colorbar,
                                                  cmap=cmap,
                                                  show_xlabel=show_xlabel,
                                                  show_ylabel=show_ylabel)
            else:
                return self.visualize_2d_surface(fig=fig, ax=ax,
                                                 figsize=figsize,
                                                 colorbar=colorbar,
                                                 cmap=cmap,
                                                 show_xlabel=show_xlabel,
                                                 show_ylabel=show_ylabel)

        elif n == 1:
            return self.visualize_1d(fig=fig, ax=ax, figsize=figsize,
                                     show_xlabel=show_xlabel,
                                     show_ylabel=show_ylabel)
        else:
            raise NotImplementedError("Distribution visualization in " +
                                      "%d dimensions is not implemented." % n)

    def visualize_2d_colormap(self, fig=None, ax=None, figsize=(8, 6),
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

    def visualize_2d_surface(self, fig=None, ax=None, figsize=(8, 6),
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

    def visualize_1d(self, fig=None, ax=None, figsize=(8, 6),
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


class WignerDistribution(Distribution):

    def __init__(self, rho=None, extent=[[-5, 5], [-5, 5]], steps=250):

        self.xvecs = [np.linspace(extent[0][0], extent[0][1], steps),
                      np.linspace(extent[1][0], extent[1][1], steps)]

        self.xlabels = [r'$\rm{Re}(\alpha)$', r'$\rm{Im}(\alpha)$']

        if rho:
            self.update(rho)

    def update(self, rho):

        self.data = wigner(rho, self.xvecs[0], self.xvecs[1])


class QDistribution(Distribution):

    def __init__(self, rho=None, extent=[[-5, 5], [-5, 5]], steps=250):

        self.xvecs = [np.linspace(extent[0][0], extent[0][1], steps),
                      np.linspace(extent[1][0], extent[1][1], steps)]

        self.xlabels = [r'$\rm{Re}(\alpha)$', r'$\rm{Im}(\alpha)$']

        if rho:
            self.update(rho)

    def update(self, rho):

        self.data = qfunc(rho, self.xvecs[0], self.xvecs[1])


class TwoModeQuadratureCorrelation(Distribution):

    def __init__(self, state=None, theta1=0.0, theta2=0.0,
                 extent=[[-5, 5], [-5, 5]], steps=250):

        self.xvecs = [np.linspace(extent[0][0], extent[0][1], steps),
                      np.linspace(extent[1][0], extent[1][1], steps)]

        self.xlabels = [r'$X_1(\theta_1)$', r'$X_2(\theta_2)$']

        self.theta1 = theta1
        self.theta2 = theta2

        if state:
            self.update(state)

    def update(self, state):
        """
        calculate probability distribution for quadrature measurement
        outcomes given a two-mode wavefunction or density matrix
        """
        if isket(state):
            self.update_psi(state)
        else:
            self.update_rho(state)

    def update_psi(self, psi):
        """
        calculate probability distribution for quadrature measurement
        outcomes given a two-mode wavefunction
        """

        X1, X2 = np.meshgrid(self.xvecs[0], self.xvecs[1])

        p = np.zeros((len(self.xvecs[0]), len(self.xvecs[1])), dtype=complex)
        N = psi.dims[0][0]

        for n1 in range(N):
            kn1 = exp(-1j * self.theta1 * n1) / \
                sqrt(sqrt(pi) * 2 ** n1 * factorial(n1)) * \
                exp(-X1 ** 2 / 2.0) * np.polyval(hermite(n1), X1)

            for n2 in range(N):
                kn2 = exp(-1j * self.theta2 * n2) / \
                    sqrt(sqrt(pi) * 2 ** n2 * factorial(n2)) * \
                    exp(-X2 ** 2 / 2.0) * np.polyval(hermite(n2), X2)
                i = state_number_index([N, N], [n1, n2])
                p += kn1 * kn2 * psi.full()[i, 0]

        self.data = abs(p) ** 2

    def update_rho(self, rho):
        """
        calculate probability distribution for quadrature measurement
        outcomes given a two-mode density matrix
        """

        X1, X2 = np.meshgrid(self.xvecs[0], self.xvecs[1])

        p = np.zeros((len(self.xvecs[0]), len(self.xvecs[1])), dtype=complex)
        N = rho.dims[0][0]

        M1 = np.zeros(
            (N, N, len(self.xvecs[0]), len(self.xvecs[1])), dtype=complex)
        M2 = np.zeros(
            (N, N, len(self.xvecs[0]), len(self.xvecs[1])), dtype=complex)

        for m in range(N):
            for n in range(N):
                M1[m, n] = exp(-1j * self.theta1 * (m - n)) / \
                    sqrt(pi * 2 ** (m + n) * factorial(n) * factorial(m)) * \
                    exp(-X1 ** 2) * np.polyval(
                        hermite(m), X1) * np.polyval(hermite(n), X1)
                M2[m, n] = exp(-1j * self.theta2 * (m - n)) / \
                    sqrt(pi * 2 ** (m + n) * factorial(n) * factorial(m)) * \
                    exp(-X2 ** 2) * np.polyval(
                        hermite(m), X2) * np.polyval(hermite(n), X2)

        for n1 in range(N):
            for n2 in range(N):
                i = state_number_index([N, N], [n1, n2])
                for p1 in range(N):
                    for p2 in range(N):
                        j = state_number_index([N, N], [p1, p2])
                        p += M1[n1, p1] * M2[n2, p2] * rho.full()[i, j]

        self.data = p


class HarmonicOscillatorWaveFunction(Distribution):

    """Calculates and represents the wave function of
       a quantum harmonic oscillator.

    The `HarmonicOscillatorWaveFunction` class computes
    the spatial distribution of the wave function for a quantum
    harmonic oscillator given a set of state coefficients (`psi`).

    By extending the `Distribution` base class, this class
    provides specialized attributes and methods tailored for modeling
    the harmonic oscillator's wave function.This implementation leverages
    the Cython function `psi_n_single_fock_multiple_position_complex`from the
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

    def __init__(self, psi=None, omega=1.0, extent=[-5, 5], steps=250):

        self.xvecs = [np.linspace(extent[0], extent[1], steps)]
        self.xlabels = [r'$x$']
        self.omega = omega

        if psi:
            self.update(psi)

    def update(self, psi):
        """
        Calculate the wavefunction for the given state of an harmonic
        oscillator
        """

        self.data = np.zeros(len(self.xvecs[0]), dtype=complex)
        N = psi.shape[0]

        for n in range(N):
            self.data += (
                psi_n_single_fock_multiple_position_complex(
                    n, self.xvecs[0].astype(complex)
                ) * psi[n, 0]
            )

        self.data *= pow(self.omega, 0.25)


class HarmonicOscillatorProbabilityFunction(Distribution):

    def __init__(self, rho=None, omega=1.0, extent=[-5, 5], steps=250):

        self.xvecs = [np.linspace(extent[0], extent[1], steps)]
        self.xlabels = [r'$x$']
        self.omega = omega

        if rho:
            self.update(rho)

    def update(self, rho):
        """
        Calculate the probability function for the given state of an harmonic
        oscillator (as density matrix)
        """

        if isket(rho):
            rho = ket2dm(rho)

        self.data = np.zeros(len(self.xvecs[0]), dtype=complex)
        M, N = rho.shape

        for m in range(M):
            k_m = pow(self.omega / pi, 0.25) / \
                sqrt(2 ** m * factorial(m)) * \
                exp(-self.xvecs[0] ** 2 / 2.0) * \
                np.polyval(hermite(m), self.xvecs[0])

            for n in range(N):
                k_n = pow(self.omega / pi, 0.25) / \
                    sqrt(2 ** n * factorial(n)) * \
                    exp(-self.xvecs[0] ** 2 / 2.0) * \
                    np.polyval(hermite(n), self.xvecs[0])

                self.data += np.conjugate(k_n) * k_m * rho.full()[m, n]
