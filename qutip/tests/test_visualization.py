import pytest
import qutip
import matplotlib as mpl


@pytest.mark.parametrize('rho, x_basis, y_basis, color_style,\
           label_top, cmap, colorbar, fig, ax', [
    (qutip.rand_dm(5), None, None, "scaled", True, None, None, None, None),
    (qutip.operator_to_vector(qutip.rand_dm(5)),
     None, None, "scaled", True, None, None, None, None),
    (qutip.operator_to_vector(qutip.rand_dm(5)).dag(),
     None, None, "scaled", True, None, None, None, None),
    (qutip.spre(qutip.rand_dm(4)),
     None, None, "scaled", True, None, None, None, None),
    (qutip.rand_dm(5),
     [1, 2, 3, 4, 5], None, "scaled", True, None, None, None, None),
    (qutip.rand_dm(5),
     None, [1, 2, 3, 4, 5], "scaled", True, None, None, None, None),
    (qutip.rand_dm(5), None, None, "scaled", True, None, None, None, None),
    (qutip.rand_dm(5), None, None, "threshold", True, None, None, None, None),
    (qutip.rand_dm(5), None, None, "phase", True, None, None, None, None),
])
def test_hinton(rho, x_basis, y_basis, color_style,
                label_top, cmap, colorbar, fig, ax):
    fig, ax = qutip.hinton(rho, x_basis, y_basis, color_style,
                           label_top, cmap=cmap, colorbar=colorbar,
                           fig=fig, ax=ax)

    assert isinstance(fig, mpl.figure.Figure)
    assert isinstance(ax, mpl.axes.Axes)


@pytest.mark.parametrize('rho, x_basis, y_basis, color_style,\
           label_top, cmap, colorbar, fig, ax, expected', [
    (qutip.basis(2, 0), None, None, "scaled", True, None, None,
     None, None, "Input quantum object must be an operator or superoperator."),
    (qutip.rand_dm(5), None, None, "color_style", True, None, None,
     None, None, "Unknown color style color_style for Hinton diagrams."),
    (qutip.spre(qutip.rand_dm(5)),
     None, None, "scaled", True, None, None, None, None,
     "Hinton plots of superoperators are currently only supported for qubits.")
])
def test_hinton_ValueError(rho, x_basis, y_basis, color_style,
                           label_top, cmap, colorbar, fig, ax, expected):
    with pytest.raises(ValueError) as exc_info:
        fig, ax = qutip.hinton(rho, x_basis, y_basis, color_style,
                               label_top, cmap=cmap, colorbar=colorbar,
                               fig=fig, ax=ax)
    assert str(exc_info.value) == expected
