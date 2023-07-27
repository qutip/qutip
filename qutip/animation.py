"""
Functions for creating animations of results of quantum dynamics simulations,
visualizations of quantum states and processes.
"""

__all__ = ['anim_wigner', 'anim_matrix_histogram', 'anim_fock_distribution']

import warnings
import itertools as it
import numpy as np
from numpy import pi, array, sin, cos, angle, log2

from packaging.version import parse as parse_version

from . import (
    Qobj, isket, ket2dm, tensor, vector_to_operator, to_super, settings
)
from .core.dimensions import flatten
from .core.superop_reps import _to_superpauli, isqubitdims
from .wigner import wigner
from .matplotlib_utilities import complex_phase_cmap

try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D

    # Define a custom _axes3D function based on the matplotlib version.
    # The auto_add_to_figure keyword is new for matplotlib>=3.4.
    if parse_version(mpl.__version__) >= parse_version('3.4'):
        def _axes3D(fig, *args, **kwargs):
            ax = Axes3D(fig, *args, auto_add_to_figure=False, **kwargs)
            return fig.add_axes(ax)
    else:
        def _axes3D(*args, **kwargs):
            return Axes3D(*args, **kwargs)
except:
    pass

import os
from functools import partial
from IPython.display import HTML
from base64 import b64encode
import matplotlib.animation as animation
from qutip import plot_wigner
import matplotlib.pyplot as plt
import matplotlib as mpl
from .visualization import *


def _is_options(options, keyward):
    if options is None:
        options = dict()

    if not isinstance(options, dict):
        raise ValueError(str(keyward) + ' must be a dict')

    return options


def make_html_video(ani, save_options=None):
    save_options = _is_options(save_options, 'save_options')
    if 'name' not in save_options.keys():
        save_options['name'] = 'animation'
    if 'writer' not in save_options.keys():
        save_options['writer'] = None
    if 'codec' not in save_options.keys():
        save_options['codec'] = None

    ani.save(save_options['name'] + '.mp4', fps=10,
             writer=save_options['writer'], codec=save_options['codec'])
    video = open(save_options['name'] + '.mp4', "rb").read()
    video_encoded = b64encode(video).decode("ascii")
    video_tag = '<video controls src="data:video/x-m4v;base64,{0}">'.format(
        video_encoded)

    return ani, HTML(video_tag)


def anim_wigner(rhos, xvec=None, yvec=None, method='clenshaw',
                projection='2d', *, cmap=None, colorbar=False,
                fig=None, ax=None, save_options=None):
    """
    Plot the time evolution of the Wigner function for density matrices
    (or kets) hat describes an oscillator mode.

    Parameters
    ----------
    rhos : list of `qutip.Qobj`
        List of density matrices (or kets) to visualize.

    xvec : array_like, optional
        x-coordinates at which to calculate the Wigner function.

    yvec : array_like, optional
        y-coordinates at which to calculate the Wigner function.  Does not
        apply to the 'fft' method.

    method : string {'clenshaw', 'iterative', 'laguerre', 'fft'},
        default='clenshaw'
        The method used for calculating the wigner function. See the
        documentation for qutip.wigner for details.

    projection: string {'2d', '3d'}, default='2d'
        Specify whether the Wigner function is to be plotted as a
        contour graph ('2d') or surface plot ('3d').

    cmap : a matplotlib cmap instance, optional
        The colormap.

    colorbar : bool, default=False
        Whether (True) or not (False) a colorbar should be attached to the
        Wigner function graph.

    fig : a matplotlib Figure instance, optional
        The Figure canvas in which the plot will be drawn.

    ax : a matplotlib axes instance, optional
        The axes context in which the plot will be drawn.

    save_options : dict, optional
        A dictionary containing options to save the animation.

        'name' : str, default='animation'
            The output filename, e.g., :file:`mymovie.mp4`.

        'writer' : `MovieWriter` or str, optional
            A `MovieWriter` instance to use or a key that identifies a
            class to use, such as 'ffmpeg'.

        'codec' : str, optional
            The video codec to use.  Not all codecs are supported by a given
            `MovieWriter`.

    Returns
    -------
    fig, ax : tuple
        A tuple of the matplotlib figure and axes instances used to produce
        the figure.
    """

    wlim = 0
    Ws = list()
    artist_list = list()

    if projection not in ('2d', '3d'):
        raise ValueError('Unexpected value of projection keyword argument')
    fig, ax = _is_fig_and_ax(fig, ax, projection)

    for rho in rhos:
        if isket(rho):
            rho = ket2dm(rho)

        if xvec is None:
            xvec = np.linspace(-7.5, 7.5, 200)
        if yvec is None:
            yvec = np.linspace(-7.5, 7.5, 200)

        W0 = wigner(rho, xvec, yvec, method=method)

        W, yvec = W0 if isinstance(W0, tuple) else (W0, yvec)
        Ws.append(W)

        wlim = max(abs(W).max(), wlim)

    norm = mpl.colors.Normalize(-wlim, wlim)
    if cmap is None:
        cmap = mpl.cm.RdBu

    for W in Ws:
        if projection == '2d':
            cf = ax.contourf(xvec, yvec, W, 100, norm=norm,
                             cmap=cmap).collections
        else:
            X, Y = np.meshgrid(xvec, yvec)
            cf = [ax.plot_surface(X, Y, W, rstride=5, cstride=5, linewidth=0.5,
                                  norm=norm, cmap=cmap)]
        artist_list.append(cf)

    ax.set_xlabel(r'$\rm{Re}(\alpha)$', fontsize=12)
    ax.set_ylabel(r'$\rm{Im}(\alpha)$', fontsize=12)

    if colorbar:
        if projection == '2d':
            shrink = 1
        else:
            shrink = .75
        cax, kw = mpl.colorbar.make_axes(ax, shrink=shrink, pad=.1)
        cbar = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)

    ax.set_title("Wigner function", fontsize=12)

    ani = animation.ArtistAnimation(fig, artist_list, interval=50,
                                    blit=True, repeat_delay=1000)
    plt.close()

    ani, html_video = make_html_video(ani, save_options)

    return fig, ani, html_video


def anim_matrix_histogram(Ms, x_basis=None, y_basis=None, limits=None,
                          bar_style='real', color_limits=None,
                          color_style='real', options=None, *,
                          cmap=None, colorbar=True, fig=None, ax=None,
                          save_options=None):
    """
    Draw a histogram for the matrix M, with the given x and y labels and title.

    Parameters
    ----------
    Ms : list of matrices of Qobj
        list of matrices to visualize

    x_basis : list of strings, optional
        list of x ticklabels

    y_basis : list of strings, optional
        list of y ticklabels

    limits : list/array with two float numbers, optional
        The z-axis limits [min, max]

    bar_style : string, default="real"

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

    color_style : string, default="real"
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

    colorbar : bool, default=True
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

        'bars_spacing' : float, default=0.1
            spacing between bars.

        'bars_alpha' : float, default=1.
            transparency of bars, should be in range 0 - 1

        'bars_lw' : float, default=0.5
            linewidth of bars' edges.

        'bars_edgecolor' : color, default='k'
            The colors of the bars' edges.
            Examples: 'k', (0.1, 0.2, 0.5) or '#0f0f0f80'.

        'shade' : bool, default=True
            Whether to shade the dark sides of the bars (True) or not (False).
            The shading is relative to plot's source of light.

        'azim' : float, default=-35
            The azimuthal viewing angle.

        'elev' : float, default=35
            The elevation viewing angle.

        'stick' : bool, default=False
            Changes xlim and ylim in such a way that bars next to
            XZ and YZ planes will stick to those planes.
            This option has no effect if ``ax`` is passed as a parameter.

        'cbar_pad' : float, default=0.04
            The fraction of the original axes between the colorbar
            and the new image axes.
            (i.e. the padding between the 3D figure and the colorbar).

        'cbar_to_z' : bool, default=False
            Whether to set the color of maximum and minimum z-values to the
            maximum and minimum colors in the colorbar (True) or not (False).

        'threshold': float, optional
            Threshold for when bars of smaller height should be transparent. If
            not set, all bars are colored according to the color map.

    save_options : dict, optional
        A dictionary containing options to save the animation.

        'name' : str, default='animation'
            The output filename, e.g., :file:`mymovie.mp4`.

        'writer' : `MovieWriter` or str, optional
            A `MovieWriter` instance to use or a key that identifies a
            class to use, such as 'ffmpeg'.

        'codec' : str, optional
            The video codec to use.  Not all codecs are supported by a given
            `MovieWriter`.

    Returns
    -------
    fig, ax : tuple
        A tuple of the matplotlib figure and axes instances used to produce
        the figure.

    Raises
    ------
    ValueError
        Input argument is not valid.

    """

    # default options
    default_opts = {'zticks': None, 'bars_spacing': 0.2,
                    'bars_alpha': 1., 'bars_lw': 0.5, 'bars_edgecolor': 'k',
                    'shade': True, 'azim': -35, 'elev': 35, 'stick': False,
                    'cbar_pad': 0.04, 'cbar_to_z': False, 'threshold': None}

    # update default_opts from input options
    if options is None:
        options = dict()

    if isinstance(options, dict):
        # check if keys in options dict are valid
        if set(options) - set(default_opts):
            raise ValueError("invalid key(s) found in options: "
                             f"{', '.join(set(options) - set(default_opts))}")
        else:
            # updating default options
            default_opts.update(options)
            options = default_opts
    else:
        raise ValueError("options must be a dictionary")

    _bars_and_colors = list()
    for i in range(len(Ms)):
        M = Ms[i]
        if isinstance(M, Qobj):
            if x_basis is None:
                x_basis = list(_cb_labels([M.shape[0]])[0])
            if y_basis is None:
                y_basis = list(_cb_labels([M.shape[1]])[1])
            # extract matrix data from Qobj
            M = M.full()

        n = np.size(M)
        xpos, ypos = np.meshgrid(range(M.shape[0]), range(M.shape[1]))
        xpos = xpos.T.flatten() + 0.5
        ypos = ypos.T.flatten() + 0.5
        zpos = np.zeros(n)
        dx = dy = (1 - options['bars_spacing']) * np.ones(n)

        bar_M = _get_matrix_components(bar_style, M, 'bar_style')

        if isinstance(limits, list) and \
                len(limits) == 2:
            z_min = limits[0]
            z_max = limits[1]
        else:
            if i == 0:
                z_min = min(bar_M)
                z_max = max(bar_M)
            else:
                z_min = min(min(bar_M), z_min)
                z_max = max(max(bar_M), z_max)

            if z_min == z_max:
                z_min -= 0.1
                z_max += 0.1

        color_M = _get_matrix_components(color_style, M, 'color_style')

        if isinstance(color_limits, list) and \
                len(color_limits) == 2:
            c_min = color_limits[0]
            c_max = color_limits[1]
        else:
            if color_style == 'phase':
                c_min = -pi
                c_max = pi
            else:
                if i == 0:
                    c_min = min(color_M)
                    c_max = max(color_M)
                else:
                    c_min = min(min(color_M), c_min)
                    c_max = max(max(color_M), c_max)

            if c_min == c_max:
                c_min -= 0.1
                c_max += 0.1
        _bars_and_colors.append((bar_M, color_M))
    norm = mpl.colors.Normalize(c_min, c_max)

    if cmap is None:
        # change later
        if color_style == 'phase':
            cmap = _cyclic_cmap()
        else:
            cmap = _sequential_cmap()

    artist_list = list()
    for bar_M, color_M in _bars_and_colors:
        colors = cmap(norm(color_M))

        colors[:, 3] = options['bars_alpha']

        if options['threshold'] is not None:
            colors[:, 3] *= 1 * (bar_M >= options['threshold'])

            idx, = np.where(bar_M < options['threshold'])
            bar_M[idx] = 0

        fig, ax = _is_fig_and_ax(fig, ax, projection='3d')

        artist = ax.bar3d(xpos, ypos, zpos, dx, dy, bar_M, color=colors,
                          edgecolors=options['bars_edgecolor'],
                          linewidths=options['bars_lw'],
                          shade=options['shade'])
        artist_list.append([artist])

    # remove vertical lines on xz and yz plane
    ax.yaxis._axinfo["grid"]['linewidth'] = 0
    ax.xaxis._axinfo["grid"]['linewidth'] = 0

    # x axis
    _update_xaxis(options['bars_spacing'], M, ax, x_basis)

    # y axis
    _update_yaxis(options['bars_spacing'], M, ax, y_basis)

    # z axis
    _update_zaxis(ax, z_min, z_max, options['zticks'])

    # stick to xz and yz plane
    _stick_to_planes(options['stick'],
                     options['azim'], ax, M,
                     options['bars_spacing'])
    ax.view_init(azim=options['azim'], elev=options['elev'])

    # color axis
    if colorbar:
        cax, kw = mpl.colorbar.make_axes(ax, shrink=.75,
                                         pad=options['cbar_pad'])
        cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm)

        if color_style == 'real':
            cb.set_label('real')
        elif color_style == 'img':
            cb.set_label('imaginary')
        elif color_style == 'abs':
            cb.set_label('absolute')
        else:
            cb.set_label('arg')
            if color_limits is None:
                cb.set_ticks([-pi, -pi / 2, 0, pi / 2, pi])
                cb.set_ticklabels(
                    (r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'))

    # removing margins
    _remove_margins(ax.xaxis)
    _remove_margins(ax.yaxis)
    _remove_margins(ax.zaxis)

    ani = animation.ArtistAnimation(fig, artist_list, interval=50,
                                    blit=True, repeat_delay=1000)
    plt.close()

    ani, html_video = make_html_video(ani, save_options)
    print(z_min, z_max)
    return fig, ani, html_video


def anim_fock_distribution(rhos, fock_numbers=None, color="green",
                           unit_y_range=True, *, fig=None, ax=None,
                           save_options=None):
    """
    Plot the Fock distribution for a density matrix (or ket) that describes
    an oscillator mode.

    Parameters
    ----------
    rhos : list of `qutip.Qobj`
        The density matrices (or kets) to visualize.

    fock_numbers : list of strings, optional
        list of x ticklabels to represent fock numbers

    color : color or list of colors, default="green"
        The colors of the bar faces.

    unit_y_range : bool, default=True
        Set y-axis limits [0, 1] or not

    fig : a matplotlib Figure instance, optional
        The Figure canvas in which the plot will be drawn.

    ax : a matplotlib axes instance, optional
        The axes context in which the plot will be drawn.

    save_options : dict, optional
        A dictionary containing options to save the animation.

        'name' : str, default='animation'
            The output filename, e.g., :file:`mymovie.mp4`.

        'writer' : `MovieWriter` or str, optional
            A `MovieWriter` instance to use or a key that identifies a
            class to use, such as 'ffmpeg'.

        'codec' : str, optional
            The video codec to use.  Not all codecs are supported by a given
            `MovieWriter`.

    Returns
    -------
    fig, ax : tuple
        A tuple of the matplotlib figure and axes instances used to produce
        the figure.
    """

    fig, ax = _is_fig_and_ax(fig, ax)

    artist_list = list()
    for rho in rhos:
        if isket(rho):
            rho = ket2dm(rho)

        N = rho.shape[0]

        artist = ax.bar(np.arange(N), np.real(rho.diag()),
                        color=color, alpha=0.6, width=0.8).patches
        artist_list.append(artist)

    if fock_numbers:
        _set_ticklabels(ax, fock_numbers, np.arange(N), 'x', fontsize=12)

    if unit_y_range:
        ax.set_ylim(0, 1)
    ax.set_xlim(-.5, N)
    ax.set_xlabel('Fock number', fontsize=12)
    ax.set_ylabel('Occupation probability', fontsize=12)

    ani = animation.ArtistAnimation(fig, artist_list, interval=50,
                                    blit=True, repeat_delay=1000)
    plt.close()

    ani, html_video = make_html_video(ani, save_options)

    return fig, ani, html_video
