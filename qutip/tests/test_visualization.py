import pytest
import qutip
import matplotlib as mpl


def to_oper_bra(oper):
    return qutip.operator_to_vector(oper).dag()


def to_oper(oper):
    return oper


@pytest.mark.parametrize('transform, args', [
    (to_oper, {}),
    (qutip.operator_to_vector, {}),
    (to_oper_bra, {}),
    (qutip.spre, {}),
    (to_oper, {'x_basis': [0, 1, 2, 3]}),
    (to_oper, {'y_basis': [0, 1, 2, 3]}),
    (to_oper, {'color_style': 'threshold'}),
    (to_oper, {'color_style': 'phase'}),
    (to_oper, {'colorbar': False}),
])
def test_hinton(transform, args):
    rho = transform(qutip.rand_dm(4))
    fig, ax = qutip.hinton(rho, **args)

    assert isinstance(fig, mpl.figure.Figure)
    assert isinstance(ax, mpl.axes.Axes)


@pytest.mark.parametrize('transform, args, error_message', [
    (to_oper, {},
     "Input quantum object must be an operator or superoperator.")
])
def test_hinton_ValueError0(transform, args, error_message):
    rho = transform(qutip.basis(2, 0))
    with pytest.raises(ValueError) as exc_info:
        fig, ax = qutip.hinton(rho, **args)
    assert str(exc_info.value) == error_message


@pytest.mark.parametrize('transform, args, error_message', [
    (to_oper, {'color_style': 'color_style'},
     "Unknown color style color_style for Hinton diagrams."),
    (qutip.spre, {},
     "Hinton plots of superoperators are currently only supported for qubits.")
])
def test_hinton_ValueError1(transform, args, error_message):
    rho = transform(qutip.rand_dm(5))
    with pytest.raises(ValueError) as exc_info:
        fig, ax = qutip.hinton(rho, **args)
    assert str(exc_info.value) == error_message
