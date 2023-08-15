"""
Functions to animate results of quantum dynamics simulations,
"""
__all__ = ['anim_wigner_sphere']

from . import plot_wigner_sphere

try:
    import matplotlib.pyplot as plt
except Exception:
    pass


def anim_wigner_sphere(wigners, reflections=False, *, cmap=None,
                       colorbar=True, fig=None, ax=None):
    """Animate a coloured Bloch sphere.

    Parameters
    ----------
    wigner : a list of transformations.
        The wigner transformation at `steps` different theta and phi.

    reflections : bool, default=False
        If the reflections of the sphere should be plotted as well.

    cmap : a matplotlib colormap instance, optional
        Color map to use when plotting.

    colorbar : bool, default=True
        Whether (True) or not (False) a colorbar should be attached.

    fig : a matplotlib Figure instance, optional
        The Figure canvas in which the plot will be drawn.

    ax : a matplotlib axes instance, optional
        The ax context in which the plot will be drawn.

    Returns
    -------
    fig, output : tuple
        A tuple of the matplotlib figure and the animation instance
        used to produce the figure.

    Notes
    -----
    Special thanks to Russell P Rundle for writing this function.
    """

    fig, ani = plot_wigner_sphere(wigners, reflections, cmap=cmap,
                                  colorbar=colorbar, fig=fig, ax=ax)
    plt.close(fig)

    return fig, ani
