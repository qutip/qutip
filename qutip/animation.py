"""
Functions to animate results of quantum dynamics simulations,
"""
__all__ = ['anim_wigner_sphere', 'anim_hinton', 'anim_sphereplot',
           'anim_matrix_histogram', 'anim_fock_distribution', 'anim_wigner',
           'anim_spin_distribution', 'anim_qubism', 'anim_schmidt']

from . import (plot_wigner_sphere, hinton, sphereplot, matrix_histogram,
               plot_fock_distribution, plot_wigner, plot_spin_distribution,
               plot_qubism, plot_schmidt)
from .solver import Result
from numpy import sqrt


def _result_state(obj):
    if isinstance(obj, Result):
        obj = obj.states
        if len(obj) == 0:
            raise ValueError('Nothing to visualize. You might have forgotten '
                             'to set options={"store_states": True}.')

    return obj


def anim_wigner_sphere(wigners, reflections=False, *, cmap=None,
                       colorbar=True, fig=None, ax=None):
    """Animate a coloured Bloch sphere.

    Parameters
    ----------
    wigners : list of transformations
        The wigner transformation at `steps` different theta and phi.

    reflections : bool, default: False
        If the reflections of the sphere should be plotted as well.

    cmap : a matplotlib colormap instance, optional
        Color map to use when plotting.

    colorbar : bool, default: True
        Whether (True) or not (False) a colorbar should be attached.

    fig : a matplotlib Figure instance, optional
        The Figure canvas in which the plot will be drawn.

    ax : a matplotlib axes instance, optional
        The ax context in which the plot will be drawn.

    Returns
    -------
    fig, ani : tuple
        A tuple of the matplotlib figure and the animation instance
        used to produce the figure.

    Notes
    -----
    Special thanks to Russell P Rundle for writing this function.
    """

    fig, ani = plot_wigner_sphere(wigners, reflections, cmap=cmap,
                                  colorbar=colorbar, fig=fig, ax=ax)

    return fig, ani


def anim_hinton(rhos, x_basis=None, y_basis=None, color_style="scaled",
                label_top=True, *, cmap=None, colorbar=True,
                fig=None, ax=None):
    """Draws an animation of Hinton diagram.

    Parameters
    ----------
    rhos : :class:`.Result` or list of :class:`.Qobj`
        Input density matrix or superoperator.

        .. note::

            Hinton plots of superoperators are currently only
            supported for qubits.

    x_basis : list of strings, optional
        list of x ticklabels to represent x basis of the input.

    y_basis : list of strings, optional
        list of y ticklabels to represent y basis of the input.

    color_style : str, {"scaled", "threshold", "phase"}, default: "scaled"

        Determines how colors are assigned to each square:

        -  If set to ``"scaled"`` (default), each color is chosen by
           passing the absolute value of the corresponding matrix
           element into `cmap` with the sign of the real part.
        -  If set to ``"threshold"``, each square is plotted as
           the maximum of `cmap` for the positive real part and as
           the minimum for the negative part of the matrix element;
           note that this generalizes `"threshold"` to complex numbers.
        -  If set to ``"phase"``, each color is chosen according to
           the angle of the corresponding matrix element.

    label_top : bool, default: True
        If True, x ticklabels will be placed on top, otherwise
        they will appear below the plot.

    cmap : a matplotlib colormap instance, optional
        Color map to use when plotting.

    colorbar : bool, default: True
        Whether (True) or not (False) a colorbar should be attached.

    fig : a matplotlib Figure instance, optional
        The Figure canvas in which the plot will be drawn.

    ax : a matplotlib axes instance, optional
        The ax context in which the plot will be drawn.

    Returns
    -------
    fig, ani : tuple
        A tuple of the matplotlib figure and the animation instance
        used to produce the figure.

    Raises
    ------
    ValueError
        Input argument is not a quantum object.

    """

    rhos = _result_state(rhos)

    fig, ani = hinton(rhos, x_basis, y_basis, color_style, label_top,
                      cmap=cmap, colorbar=colorbar, fig=fig, ax=ax)

    return fig, ani


def anim_sphereplot(V, theta, phi, *, cmap=None,
                    colorbar=True, fig=None, ax=None):
    """animation of a matrices of values on a sphere

    Parameters
    ----------
    V : list of array instances
        Data set to be plotted

    theta : float
        Angle with respect to z-axis. Its range is between 0 and pi

    phi : float
        Angle in x-y plane. Its range is between 0 and 2*pi

    cmap : a matplotlib colormap instance, optional
        Color map to use when plotting.

    colorbar : bool, default: True
        Whether (True) or not (False) a colorbar should be attached.

    fig : a matplotlib Figure instance, optional
        The Figure canvas in which the plot will be drawn.

    ax : a matplotlib axes instance, optional
        The axes context in which the plot will be drawn.

    Returns
    -------
    fig, ani : tuple
        A tuple of the matplotlib figure and the animation instance
        used to produce the figure.
    """

    fig, ani = sphereplot(V, theta, phi, cmap=cmap,
                          colorbar=colorbar, fig=fig, ax=ax)

    return fig, ani


def anim_matrix_histogram(Ms, x_basis=None, y_basis=None, limits=None,
                          bar_style='real', color_limits=None,
                          color_style='real', options=None, *, cmap=None,
                          colorbar=True, fig=None, ax=None):
    """
    Draw an animation of a histogram for the matrix M,
    with the given x and y labels.

    Parameters
    ----------
    Ms : list of matrices or :class:`.Result`
        The matrix to visualize

    x_basis : list of strings, optional
        list of x ticklabels

    y_basis : list of strings, optional
        list of y ticklabels

    limits : list/array with two float numbers, optional
        The z-axis limits [min, max]

    bar_style : str, {"real", "img", "abs", "phase"}, default: "real"

        -  If set to ``"real"`` (default), each bar is plotted
           as the real part of the corresponding matrix element
        -  If set to ``"img"``, each bar is plotted
           as the imaginary part of the corresponding matrix element
        -  If set to ``"abs"``, each bar is plotted
           as the absolute value of the corresponding matrix element
        -  If set to ``"phase"`` (default), each bar is plotted
           as the angle of the corresponding matrix element

    color_limits : list/array with two float numbers, optional
        The limits of colorbar [min, max]

    color_style : str, {"real", "img", "abs", "phase"}, default: "real"
        Determines how colors are assigned to each square:

        -  If set to ``"real"`` (default), each color is chosen
           according to the real part of the corresponding matrix element.
        -  If set to ``"img"``, each color is chosen according to
           the imaginary part of the corresponding matrix element.
        -  If set to ``"abs"``, each color is chosen according to
           the absolute value of the corresponding matrix element.
        -  If set to ``"phase"``, each color is chosen according to
           the angle of the corresponding matrix element.

    cmap : a matplotlib colormap instance, optional
        Color map to use when plotting.

    colorbar : bool, default: True
        show colorbar

    fig : a matplotlib Figure instance, optional
        The Figure canvas in which the plot will be drawn.

    ax : a matplotlib axes instance, optional
        The axes context in which the plot will be drawn.

    options : dict, optional
        A dictionary containing extra options for the plot.
        The names (keys) and values of the options are
        described below:

        'zticks' : list of numbers, optional
            A list of z-axis tick locations.

        'bars_spacing' : float, default: 0.1
            spacing between bars.

        'bars_alpha' : float, default: 1.
            transparency of bars, should be in range 0 - 1

        'bars_lw' : float, default: 0.5
            linewidth of bars' edges.

        'bars_edgecolor' : color, default: 'k'
            The colors of the bars' edges.
            Examples: 'k', (0.1, 0.2, 0.5) or '#0f0f0f80'.

        'shade' : bool, default: True
            Whether to shade the dark sides of the bars (True) or not (False).
            The shading is relative to plot's source of light.

        'azim' : float, default: -35
            The azimuthal viewing angle.

        'elev' : float, default: 35
            The elevation viewing angle.

        'stick' : bool, default: False
            Changes xlim and ylim in such a way that bars next to
            XZ and YZ planes will stick to those planes.
            This option has no effect if ``ax`` is passed as a parameter.

        'cbar_pad' : float, default: 0.04
            The fraction of the original axes between the colorbar
            and the new image axes.
            (i.e. the padding between the 3D figure and the colorbar).

        'cbar_to_z' : bool, default: False
            Whether to set the color of maximum and minimum z-values to the
            maximum and minimum colors in the colorbar (True) or not (False).

        'threshold': float, optional
            Threshold for when bars of smaller height should be transparent. If
            not set, all bars are colored according to the color map.

    Returns
    -------
    fig, ani : tuple
        A tuple of the matplotlib figure and the animation instance
        used to produce the figure.

    Raises
    ------
    ValueError
        Input argument is not valid.

    """

    Ms = _result_state(Ms)

    fig, ani = matrix_histogram(Ms, x_basis, y_basis, limits, bar_style,
                                color_limits, color_style, options, cmap=cmap,
                                colorbar=colorbar, fig=fig, ax=ax)

    return fig, ani


def anim_fock_distribution(rhos, fock_numbers=None, color="green",
                           unit_y_range=True, *, fig=None, ax=None):
    """
    Animation of the Fock distribution for a density matrix (or ket)
    that describes an oscillator mode.

    Parameters
    ----------
    rhos : :class:`.Result` or list of :class:`.Qobj`
        The density matrix (or ket) of the state to visualize.

    fock_numbers : list of strings, optional
        list of x ticklabels to represent fock numbers

    color : color or list of colors, default: "green"
        The colors of the bar faces.

    unit_y_range : bool, default: True
        Set y-axis limits [0, 1] or not

    fig : a matplotlib Figure instance, optional
        The Figure canvas in which the plot will be drawn.

    ax : a matplotlib axes instance, optional
        The axes context in which the plot will be drawn.

    Returns
    -------
    fig, ani : tuple
        A tuple of the matplotlib figure and the animation instance
        used to produce the figure.
    """

    rhos = _result_state(rhos)

    fig, ani = plot_fock_distribution(rhos, fock_numbers, color,
                                      unit_y_range, fig=fig, ax=ax)

    return fig, ani


def anim_wigner(rhos, xvec=None, yvec=None, method='clenshaw', projection='2d',
                g=sqrt(2), sparse=False, parfor=False, *,
                cmap=None, colorbar=False, fig=None, ax=None):
    """
    Animation of the Wigner function for a density matrix (or ket)
    that describes an oscillator mode.

    Parameters
    ----------
    rhos : :class:`.Result` or list of :class:`.Qobj`
        The density matrix (or ket) of the state to visualize.

    xvec : array_like, optional
        x-coordinates at which to calculate the Wigner function.

    yvec : array_like, optional
        y-coordinates at which to calculate the Wigner function.  Does not
        apply to the 'fft' method.

    method : str {'clenshaw', 'iterative', 'laguerre', 'fft'}, default: 'clenshaw'
        The method used for calculating the wigner function. See the
        documentation for qutip.wigner for details.

    projection: str {'2d', '3d'}, default: '2d'
        Specify whether the Wigner function is to be plotted as a
        contour graph ('2d') or surface plot ('3d').

    g : float
        Scaling factor for `a = 0.5 * g * (x + iy)`, default `g = sqrt(2)`.
        See the documentation for qutip.wigner for details.

    sparse : bool {False, True}
        Flag for sparse format.
        See the documentation for qutip.wigner for details.

    parfor : bool {False, True}
        Flag for parallel calculation.
        See the documentation for qutip.wigner for details.

    cmap : a matplotlib cmap instance, optional
        The colormap.

    colorbar : bool, default: False
        Whether (True) or not (False) a colorbar should be attached to the
        Wigner function graph.

    fig : a matplotlib Figure instance, optional
        The Figure canvas in which the plot will be drawn.

    ax : a matplotlib axes instance, optional
        The axes context in which the plot will be drawn.

    Returns
    -------
    fig, ani : tuple
        A tuple of the matplotlib figure and the animation instance
        used to produce the figure.
    """

    rhos = _result_state(rhos)

    fig, ani = plot_wigner(rhos, xvec, yvec, method=method, g=g, sparse=sparse,
                           parfor=parfor, projection=projection,
                           cmap=cmap, colorbar=colorbar, fig=fig, ax=ax)

    return fig, ani


def anim_spin_distribution(Ps, THETA, PHI, projection='2d', *,
                           cmap=None, colorbar=False, fig=None, ax=None):
    """
    Animation of a spin distribution (given as meshgrid data).

    Parameters
    ----------
    Ps : list of matrices
        Distribution values as a meshgrid matrix.

    THETA : matrix
        Meshgrid matrix for the theta coordinate. Its range is between 0 and pi

    PHI : matrix
        Meshgrid matrix for the phi coordinate. Its range is between 0 and 2*pi

    projection: str {'2d', '3d'}, default: '2d'
        Specify whether the spin distribution function is to be plotted as a 2D
        projection where the surface of the unit sphere is mapped on
        the unit disk ('2d') or surface plot ('3d').

    cmap : a matplotlib cmap instance, optional
        The colormap.

    colorbar : bool, default: False
        Whether (True) or not (False) a colorbar should be attached to the
        Wigner function graph.

    fig : a matplotlib figure instance, optional
        The figure canvas on which the plot will be drawn.

    ax : a matplotlib axis instance, optional
        The axis context in which the plot will be drawn.

    Returns
    -------
    fig, ani : tuple
        A tuple of the matplotlib figure and the animation instance
        used to produce the figure.
    """

    fig, ani = plot_spin_distribution(Ps, THETA, PHI, projection, cmap=cmap,
                                      colorbar=colorbar, fig=fig, ax=ax)

    return fig, ani


def anim_qubism(kets, theme='light', how='pairs', grid_iteration=1,
                legend_iteration=0, *, fig=None, ax=None):
    """
    Animation of Qubism plot for pure states of many qudits.
    Works best for spin chains, especially with even number of particles
    of the same dimension.  Allows to see entanglement between first
    2k particles and the rest.

    .. note::

        colorblind_safe does not apply because of its unique colormap

    Parameters
    ----------
    kets : :class:`.Result` or list of :class:`.Qobj`
        Pure states for animation.

    theme : str {'light', 'dark'}, default: 'light'
        Set coloring theme for mapping complex values into colors.
        See: complex_array_to_rgb.

    how : str {'pairs', 'pairs_skewed', 'before_after'}, default: 'pairs'
        Type of Qubism plotting.  Options:

        - 'pairs' - typical coordinates,
        - 'pairs_skewed' - for ferromagnetic/antriferromagnetic plots,
        - 'before_after' - related to Schmidt plot (see also: plot_schmidt).

    grid_iteration : int, default: 1
        Helper lines to be drawn on plot.
        Show tiles for 2*grid_iteration particles vs all others.

    legend_iteration : int or 'grid_iteration' or 'all', default: 0
        Show labels for first ``2*legend_iteration`` particles.  Option
        'grid_iteration' sets the same number of particles as for
        grid_iteration.  Option 'all' makes label for all particles.  Typically
        it should be 0, 1, 2 or perhaps 3.

    fig : a matplotlib figure instance, optional
        The figure canvas on which the plot will be drawn.

    ax : a matplotlib axis instance, optional
        The axis context in which the plot will be drawn.

    Returns
    -------
    fig, ani : tuple
        A tuple of the matplotlib figure and the animation instance
        used to produce the figure.

    Notes
    -----
    See also [1]_.

    References
    ----------
    .. [1] J. Rodriguez-Laguna, P. Migdal, M. Ibanez Berganza, M. Lewenstein
       and G. Sierra, *Qubism: self-similar visualization of many-body
       wavefunctions*, `New J. Phys. 14 053028
       <https://dx.doi.org/10.1088/1367-2630/14/5/053028>`_, arXiv:1112.3560
       (2012), open access.
    """

    kets = _result_state(kets)

    fig, ani = plot_qubism(kets, theme, how, grid_iteration,
                           legend_iteration, fig=fig, ax=ax)

    return fig, ani


def anim_schmidt(kets, theme='light', splitting=None,
                 labels_iteration=(3, 2), *, fig=None, ax=None):
    """
    Animation of Schmidt decomposition.
    Converts a state into a matrix (A_ij -> A_i^j),
    where rows are first particles and columns - last.

    See also: plot_qubism with how='before_after' for a similar plot.

    .. note::

        colorblind_safe does not apply because of its unique colormap

    Parameters
    ----------
    ket : :class:`.Result` or list of :class:`.Qobj`
        Pure states for animation.

    theme : str {'light', 'dark'}, default: 'light'
        Set coloring theme for mapping complex values into colors.
        See: complex_array_to_rgb.

    splitting : int, optional
        Plot for a number of first particles versus the rest.
        If not given, it is (number of particles + 1) // 2.

    labels_iteration : int or pair of ints, default: (3, 2)
        Number of particles to be shown as tick labels,
        for first (vertical) and last (horizontal) particles, respectively.

    fig : a matplotlib figure instance, optional
        The figure canvas on which the plot will be drawn.

    ax : a matplotlib axis instance, optional
        The axis context in which the plot will be drawn.

    Returns
    -------
    fig, ani : tuple
        A tuple of the matplotlib figure and the animation instance
        used to produce the figure.

    """

    kets = _result_state(kets)

    fig, ani = plot_schmidt(kets, theme, splitting, labels_iteration,
                            fig=fig, ax=ax)

    return fig, ani
