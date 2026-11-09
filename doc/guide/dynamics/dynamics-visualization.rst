.. _dynamics-visualization:

**********************************
Visualising Solver Results
**********************************

All QuTiP solvers return result objects that include built-in plotting
methods.  These make it easy to quickly inspect expectation values
without writing boilerplate matplotlib code.

Basic Usage
===========

After running any solver, call :meth:`~qutip.solver.result.Result.plot_expect`
on the result:

.. plot::
    :context: reset

    import qutip
    import numpy as np

    H = qutip.num(5)
    psi0 = qutip.basis(5, 4)
    tlist = np.linspace(0, 10, 100)
    a = qutip.destroy(5)

    result = qutip.mesolve(H, psi0, tlist, [0.5 * a],
                           e_ops=[a.dag() * a])
    result.plot_expect()

Dictionary Labels
=================

When ``e_ops`` is passed as a dictionary, the keys are used as legend labels
automatically:

.. plot::
    :context: close-figs

    result = qutip.mesolve(H, psi0, tlist, [0.5 * a],
                           e_ops={"photons": a.dag() * a,
                                  "field": a + a.dag()})
    result.plot_expect()

Choosing What to Plot
=====================

The ``which`` option selects a subset of the ``e_ops``, by key or by position:

.. plot::
    :context: close-figs

    result.plot_expect(which="photons")

Several can be given as a list, e.g. ``which=["photons", "field"]`` or
``which=[0, 1]``.

Labels and Colors
=================

Use ``labels`` and ``colors`` to override the defaults.  Each takes one
entry per plotted curve:

.. plot::
    :context: close-figs

    result.plot_expect(
        labels=["Photon number", "Field quadrature"],
        colors=["tab:red", "tab:blue"],
        title="Damped oscillator",
    )

Any other keyword argument is passed on to matplotlib's ``plot``, for example
``linestyle`` or ``linewidth``:

.. plot::
    :context: close-figs

    result.plot_expect(linestyle="--", linewidth=1)

Separate Subplots
=================

To plot each expectation value in its own subplot, use ``separate_axes=True``:

.. plot::
    :context: close-figs

    result.plot_expect(separate_axes=True)


Comparing Several Results
=========================

Pass the same ``axes`` to several calls to draw the results on one plot.
Give each a label and a color to tell them apart:

.. plot::
    :context: close-figs

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for rate, color in zip([0.2, 0.5, 1.0], ["r", "g", "b"]):
        result = qutip.mesolve(H, psi0, tlist, [rate * a],
                               e_ops=[a.dag() * a])
        result.plot_expect(
            axes=ax,
            labels=[f"rate = {rate}"],
            colors=[color],
            title="Photon number for different decay rates",
        )


Monte Carlo Trajectories
=========================

Results from :func:`~qutip.solver.mcsolve.mcsolve` and other trajectory solvers
(:class:`~qutip.solver.multitrajresult.MultiTrajResult`) average over many
trajectories with :meth:`~qutip.solver.MultiTrajResult.plot_expect`:

.. plot::
    :context: close-figs

    result = qutip.mcsolve(H, psi0, tlist, [0.5 * a],
                           e_ops=[a.dag() * a],
                           ntraj=50,
                           options={"keep_runs_results": True})

    result.plot_expect(show_error_bar=True)


Each error bars is the error of the average, not the spread of individual trajectories.

Individual trajectories can be overlaid on the average.  This requires
``keep_runs_results=True``:

.. plot::
    :context: close-figs

    # Show 10 trajectories behind the average
    result.plot_expect(show_trajectories=10)


Trajectory styling can be customised.  By default, trajectories use the color
of their curve:


.. plot::
    :context: close-figs

    result.plot_expect(
        show_average=False,
        show_trajectories=10,
        trajectory_kwargs={"color": "blue", "alpha": 0.1},
    )


Spread of the Trajectories
--------------------------

The standard deviation over trajectories can be shown as a shaded band, and
percentile lines can be drawn. The percentile can be obtain assuming a normal
distribution ``percentile_source="std"`` (default) or actual data
``percentile_source="traj"``:

.. plot::
    :context: close-figs

    result.plot_expect(
        show_std_band=True,
        show_percentile=[0.05, 0.95],
        percentile_source="traj"
    )



Photocurrent
============

Monte Carlo results also support plotting the
:meth:`photocurrent <.McResult.plot_photocurrent>`:

.. plot::
    :context: close-figs

    result.plot_photocurrent()

The same options as for ``plot_expect`` are available: ``which``, ``labels``,
``colors``, ``separate_axes``, ``show_trajectories``, and ``show_error_bar``:

.. plot::
    :context: close-figs

    result.plot_photocurrent(show_trajectories=5, show_error_bar=True)


Customisation
=============

All methods accept ``fig`` and ``axes`` keyword arguments for embedding
plots in custom layouts:

.. plot::
    :context: close-figs

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    result.plot_expect(axes=ax1, title="Expectation values")
    result.plot_photocurrent(axes=ax2, title="Photocurrent")
    plt.tight_layout()

Other options include ``title``, ``xlabel``, ``ylabel`` and ``show_legend``.
