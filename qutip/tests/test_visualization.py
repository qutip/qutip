import pytest
import qutip
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


@pytest.mark.parametrize('args', [
    ({'reflections': True}),
    ({'cmap': mpl.cm.cividis}),
    ({'colorbar': False}),
])
def test_plot_wigner_sphere(args):
    psi = qutip.rand_ket(5)
    wigner = qutip.wigner_transform(psi, 2, False, 50, ["x"])

    fig, ax = qutip.plot_wigner_sphere(wigner, **args)
    plt.close()

    assert isinstance(fig, mpl.figure.Figure)
    assert isinstance(ax, mpl.axes.Axes)


@pytest.mark.parametrize('args', [
    ({}),
    ({'options': {'zticks': [1]}}),
    ({'x_basis': [1, 2, 3, 4, 5]}),
    ({'y_basis': [1, 2, 3, 4, 5]}),
    ({'limits': [0, 1]}),
    ({'color_limits': [0, 1]}),
    ({'color_style': 'phase'}),
    ({'options': {'threshold': 0.1}}),
    ({'color_style': 'real', 'colorbar': True}),
    ({'color_style': 'img', 'colorbar': True}),
    ({'color_style': 'abs', 'colorbar': True}),
    ({'color_style': 'phase', 'colorbar': True}),
    ({'color_limits': [0, 1], 'color_style': 'phase', 'colorbar': True})
])
def test_matrix_histogram(args):
    rho = qutip.rand_dm(5)

    fig, ax = qutip.matrix_histogram(rho, **args)
    plt.close()

    assert isinstance(fig, mpl.figure.Figure)
    assert isinstance(ax, mpl.axes.Axes)


def test_matrix_histogram_ValueError():
    text = "options must be a dictionary"
    with pytest.raises(ValueError) as exc_info:
        fig, ax = qutip.matrix_histogram(qutip.rand_dm(5),
                                         options='error')
    assert str(exc_info.value) == text
