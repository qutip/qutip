import pytest
import qutip
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def test_cyclic():
    qutip.settings.colorblind_safe = True
    rho = qutip.rand_dm(5)
    fig, ax = qutip.hinton(rho, color_style='phase')
    plt.close()
    qutip.settings.colorblind_safe = False

    assert isinstance(fig, mpl.figure.Figure)
    assert isinstance(ax, mpl.axes.Axes)


def test_diverging():
    qutip.settings.colorblind_safe = True
    rho = qutip.rand_dm(5)
    fig, ax = qutip.hinton(rho)
    plt.close()
    qutip.settings.colorblind_safe = False

    assert isinstance(fig, mpl.figure.Figure)
    assert isinstance(ax, mpl.axes.Axes)


def test_sequential():
    qutip.settings.colorblind_safe = True
    theta = np.linspace(0, np.pi, 90)
    phi = np.linspace(0, 2 * np.pi, 60)

    fig, ax = qutip.sphereplot(theta, phi,
                               qutip.orbital(theta, phi, qutip.basis(3, 0)).T)
    plt.close()
    qutip.settings.colorblind_safe = False

    assert isinstance(fig, mpl.figure.Figure)
    assert isinstance(ax, mpl.axes.Axes)


@pytest.mark.parametrize('f, a, projection', [
    (True, True, '2d'),
    (True, True, '3d'),
    (True, False, '2d'),
    (True, False, '3d'),
    (False, True, '2d'),
    (False, True, '3d'),
    (False, False, '2d'),
    (False, False, '3d'),
])
def test_is_fig_and_ax(f, a, projection):
    rho = qutip.rand_dm(5)

    fig = plt.figure()
    ax = None
    if a:
        if projection == '2d':
            ax = fig.add_subplot(111)
        else:
            ax = fig.add_subplot(111, projection='3d')
    if not f:
        fig = None

    rho = qutip.rand_dm(5)
    fig, ax = qutip.plot_wigner(rho, projection=projection,
                                fig=fig, ax=ax)
    plt.close()
    assert isinstance(fig, mpl.figure.Figure)
    assert isinstance(ax, mpl.axes.Axes)


def test_set_ticklabels():
    rho = qutip.rand_dm(5)
    text = "got 1 ticklabels but needed 5"
    with pytest.raises(Exception) as exc_info:
        fig, ax = qutip.hinton(rho, x_basis=[1])
    assert str(exc_info.value) == text


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
    plt.close()

    assert isinstance(fig, mpl.figure.Figure)
    assert isinstance(ax, mpl.axes.Axes)


def test_hinton1():
    fig, ax = qutip.hinton(np.zeros((3, 3)))
    plt.close()

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


@pytest.mark.parametrize('args', [
    ({'cmap': mpl.cm.cividis}),
    ({'colorbar': False}),
])
def test_sphereplot(args):
    theta = np.linspace(0, np.pi, 90)
    phi = np.linspace(0, 2 * np.pi, 60)

    fig, ax = qutip.sphereplot(theta, phi,
                               qutip.orbital(theta, phi, qutip.basis(3, 0)).T,
                               **args)
    plt.close()

    assert isinstance(fig, mpl.figure.Figure)
    assert isinstance(ax, mpl.axes.Axes)


@pytest.mark.parametrize('args', [
    ({'h_labels': ['H0', 'H0+Hint']}),
    ({'energy_levels': [-2, 0, 2]}),
])
def test_plot_energy_levels(args):
    H0 = qutip.tensor(qutip.sigmaz(), qutip.identity(2)) + \
        qutip.tensor(qutip.identity(2), qutip.sigmaz())
    Hint = 0.1 * qutip.tensor(qutip.sigmax(), qutip.sigmax())

    fig, ax = qutip.plot_energy_levels([H0, Hint], **args)
    plt.close()

    assert isinstance(fig, mpl.figure.Figure)
    assert isinstance(ax, mpl.axes.Axes)


def test_plot_energy_levels_ValueError():
    with pytest.raises(ValueError) as exc_info:
        fig, ax = qutip.plot_energy_levels(1)
    assert str(exc_info.value) == "H_list must be a list of Qobj instances"


@pytest.mark.parametrize('rho_type, args', [
    ('oper', {}),
    ('ket', {}),
    ('oper', {'fock_numbers': [0, 1, 2, 3]}),
    ('oper', {'unit_y_range': False}),
])
def test_plot_fock_distribution(rho_type, args):
    if rho_type == 'oper':
        rho = qutip.rand_dm(4)
    else:
        rho = qutip.basis(2, 0)

    fig, ax = qutip.plot_fock_distribution(rho, **args)
    plt.close()

    assert isinstance(fig, mpl.figure.Figure)
    assert isinstance(ax, mpl.axes.Axes)


@pytest.mark.parametrize('rho_type, args', [
    ('oper', {}),
    ('ket', {}),
    ('oper', {'xvec': np.linspace(-1, 1, 100)}),
    ('oper', {'yvec': np.linspace(-1, 1, 100)}),
    ('oper', {'method': 'fft'}),
    ('oper', {'projection': '3d'}),
    ('oper', {'colorbar': True})
])
def test_plot_wigner(rho_type, args):
    if rho_type == 'oper':
        rho = qutip.rand_dm(4)
    else:
        rho = qutip.basis(2, 0)

    fig, ax = qutip.plot_wigner(rho, **args)
    plt.close()

    assert isinstance(fig, mpl.figure.Figure)
    assert isinstance(ax, mpl.axes.Axes)


def test_plot_wigner_ValueError():
    text = "Unexpected value of projection keyword argument"
    with pytest.raises(ValueError) as exc_info:
        rho = qutip.rand_dm(4)

        fig, ax = qutip.plot_wigner(rho, projection=1)
        plt.close()
    assert str(exc_info.value) == text


@pytest.mark.parametrize('n_of_results, n_of_e_ops, one_axes, args', [
    (1, 3, False, {}),
    (1, 3, False, {'ylabels': [1, 2, 3]}),
    (1, 1, True, {}),
    (2, 3, False, {}),
])
def test_plot_expectation_values(n_of_results, n_of_e_ops, one_axes, args):
    H = qutip.sigmaz() + 0.3 * qutip.sigmay()
    e_ops = [qutip.sigmax(), qutip.sigmay(), qutip.sigmaz()]
    times = np.linspace(0, 10, 100)
    psi0 = (qutip.basis(2, 0) + qutip.basis(2, 1)).unit()
    result = qutip.mesolve(H, psi0, times, [], e_ops[:n_of_e_ops])

    if n_of_results == 1:
        results = result
    else:
        results = [result, result]

    if one_axes:
        fig = plt.figure()
        axes = fig.add_subplot(111)
    else:
        fig = None
        axes = None

    fig, axes = qutip.plot_expectation_values(results, **args,
                                              fig=fig, axes=axes)
    plt.close()

    assert isinstance(fig, mpl.figure.Figure)
    assert isinstance(axes, np.ndarray)


@pytest.mark.filterwarnings(
    "ignore:The input coordinates to pcolor:UserWarning"
)
@pytest.mark.parametrize('color, args', [
    ('sequential', {}),
    ('diverging', {}),
    ('sequential', {'projection': '3d'}),
    ('sequential', {'colorbar': True})
])
def test_plot_spin_distribution(color, args):
    j = 5
    psi = qutip.spin_state(j, -j)
    psi = qutip.spin_coherent(j, np.random.rand() * np.pi,
                              np.random.rand() * 2 * np.pi)
    rho = qutip.ket2dm(psi)
    theta = np.linspace(0, np.pi, 50)
    phi = np.linspace(0, 2 * np.pi, 50)
    Q, THETA, PHI = qutip.spin_q_function(psi, theta, phi)
    if color == 'diverging':
        Q *= -1e12
        Q[0, 0] = -1e13

    fig, ax = qutip.plot_spin_distribution(Q, THETA, PHI, **args)
    plt.close()

    assert isinstance(fig, mpl.figure.Figure)
    assert isinstance(ax, mpl.axes.Axes)


def test_plot_spin_distribution_ValueError():
    text = "Unexpected value of projection keyword argument"
    j = 5
    psi = qutip.spin_state(j, -j)
    psi = qutip.spin_coherent(j, np.random.rand() * np.pi,
                              np.random.rand() * 2 * np.pi)
    rho = qutip.ket2dm(psi)
    theta = np.linspace(0, np.pi, 50)
    phi = np.linspace(0, 2 * np.pi, 50)
    Q, THETA, PHI = qutip.spin_q_function(psi, theta, phi)
    with pytest.raises(ValueError) as exc_info:
        fig, ax = qutip.plot_spin_distribution(Q, THETA, PHI, projection=1)
    assert str(exc_info.value) == text


@pytest.mark.parametrize('args', [
    ({}),
    ({'theme': 'dark'}),
])
def test_complex_array_to_rgb(args):
    Y = qutip.complex_array_to_rgb(np.zeros((3, 3)), **args)

    assert isinstance(Y, np.ndarray)


@pytest.mark.parametrize('dims, args', [
    (2, {}),
    (3, {}),
    (2, {'how': 'pairs'}),
    (2, {'how': 'pairs_skewed'}),
    (2, {'how': 'before_after'}),
    (2, {'legend_iteration': 'all'}),
    (2, {'legend_iteration': 'grid_iteration'}),
    (2, {'legend_iteration': 1, 'how': 'before_after'}),
    (2, {'legend_iteration': 1, 'how': 'pairs'}),
])
def test_plot_qubism(dims, args):
    if dims == 2:
        state = qutip.ket("01")
    else:
        state = qutip.ket("010")

    fig, ax = qutip.plot_qubism(state, **args)
    plt.close()

    assert isinstance(fig, mpl.figure.Figure)
    assert isinstance(ax, mpl.axes.Axes)


@pytest.mark.parametrize('ket, args, expected', [
    (False, {}, "Qubism works only for pure states, i.e. kets."),
    (True, {'how': 'error'}, "No such 'how'."),
    (True, {'legend_iteration': 'error'}, "No such option for " +
     "legend_iteration keyword argument. " +
     "Use 'all', 'grid_iteration' or an integer."),
])
def test_plot_qubism_Error(ket, args, expected):
    if ket:
        state = qutip.ket("01")
    else:
        state = qutip.bra("01")
    with pytest.raises(Exception) as exc_info:
        fig, ax = qutip.plot_qubism(state, **args)
    assert str(exc_info.value) == expected


def test_plot_qubism_mock(monkeypatch: pytest.MonkeyPatch):
    text = "For 'pairs_skewed' pairs of dimensions need to be the same."

    def mock_func(ket, kwarg):
        return ket
    monkeypatch.setattr("qutip.visualization.tensor", mock_func)
    with pytest.raises(Exception) as exc_info:
        qutip.plot_qubism(qutip.ket("010"), how='pairs_skewed')
    assert str(exc_info.value) == text


@pytest.mark.parametrize('args', [
    ({'splitting': None}),
    ({'labels_iteration': 1}),
])
def test_plot_schmidt(args):
    state = qutip.ket("01")

    fig, ax = qutip.plot_schmidt(state, **args)
    plt.close()

    assert isinstance(fig, mpl.figure.Figure)
    assert isinstance(ax, mpl.axes.Axes)


def test_plot_schmidt_Error():
    state = qutip.bra("01")
    text = "Schmidt plot works only for pure states, i.e. kets."
    with pytest.raises(Exception) as exc_info:
        fig, ax = qutip.plot_schmidt(state)
    assert str(exc_info.value) == text
