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

Separate Subplots
=================

To plot each expectation value in its own subplot, use ``separate_axes=True``:

.. plot::
    :context: close-figs

    result.plot_expect(separate_axes=True)

Monte Carlo Trajectories
=========================

For Monte Carlo results
(:class:`~qutip.solver.multitrajresult.McResult`),
individual trajectories can be overlaid on the average:

.. plot::
    :context: close-figs

    result = qutip.mcsolve(H, psi0, tlist, [0.5 * a],
                           e_ops=[a.dag() * a],
                           ntraj=50,
                           options={"keep_runs_results": True})

    # Show 10 trajectories behind the average
    result.plot_expect(show_trajectories=10)

Trajectory styling can be customised:

.. plot::
    :context: close-figs

    result.plot_expect(
        show_trajectories=10,
        trajectory_kwargs={"color": "blue", "alpha": 0.1},
    )

Photocurrent
============

Monte Carlo results also support plotting the photocurrent:

.. plot::
    :context: close-figs

    result.plot_photocurrent()

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

Other options include ``title``, ``xlabel``, ``ylabel``, ``labels``,
and ``show_legend``.