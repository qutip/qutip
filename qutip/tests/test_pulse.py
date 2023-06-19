import numpy as np
from numpy.testing import assert_,  assert_allclose

from qutip.qobj import Qobj
from qutip.qip.pulse import Pulse, Drift
from qutip.operators import sigmax, sigmay, sigmaz, identity
from qutip.tensor import tensor


def TestBasicPulse():
    """
    Test for basic pulse generation and attributes.
    """
    coeff = np.array([0.1, 0.2, 0.3, 0.4])
    tlist = np.array([0., 1., 2., 3.])
    ham = sigmaz()

    # Basic tests
    pulse1 = Pulse(ham, 1, tlist, coeff)
    assert_allclose(
        pulse1.get_ideal_qobjevo(2).ops[0].qobj, tensor(identity(2), sigmaz()))
    pulse1.tlist = 2 * tlist
    assert_allclose(pulse1.tlist, 2 * tlist)
    pulse1.tlist = tlist
    pulse1.coeff = 2 * coeff
    assert_allclose(pulse1.coeff, 2 * coeff)
    pulse1.coeff = coeff
    pulse1.qobj = 2 * sigmay()
    assert_allclose(pulse1.qobj, 2 * sigmay())
    pulse1.qobj = ham
    pulse1.targets = 3
    assert_allclose(pulse1.targets, 3)
    pulse1.targets = 1
    assert_allclose(pulse1.get_ideal_qobj(2), tensor(identity(2), sigmaz()))


def TestCoherentNoise():
    """
    Test for pulse genration with coherent noise.
    """
    coeff = np.array([0.1, 0.2, 0.3, 0.4])
    tlist = np.array([0., 1., 2., 3.])
    ham = sigmaz()
    pulse1 = Pulse(ham, 1, tlist, coeff)
    # Add coherent noise with the same tlist
    pulse1.add_coherent_noise(sigmay(), 0, tlist, coeff)
    assert_allclose(
        pulse1.get_ideal_qobjevo(2).ops[0].qobj, tensor(identity(2), sigmaz()))
    assert_(len(pulse1.coherent_noise) == 1)
    noise_qu, c_ops = pulse1.get_noisy_qobjevo(2)
    assert_allclose(c_ops, [])
    assert_allclose(noise_qu.tlist, np.array([0., 1., 2., 3.]))
    qobj_list = [ele.qobj for ele in noise_qu.ops]
    assert_(tensor(identity(2), sigmaz()) in qobj_list)
    assert_(tensor(sigmay(), identity(2)) in qobj_list)
    for ele in noise_qu.ops:
        assert_allclose(ele.coeff, coeff)


def TestNoisyPulse():
    """
    Test for lindblad noise and different tlist
    """
    coeff = np.array([0.1, 0.2, 0.3, 0.4])
    tlist = np.array([0., 1., 2., 3.])
    ham = sigmaz()
    pulse1 = Pulse(ham, 1, tlist, coeff)
    # Add coherent noise and lindblad noise with different tlist
    pulse1.spline_kind = "step_func"
    tlist_noise = np.array([1., 2.5, 3.])
    coeff_noise = np.array([0.5, 0.1, 0.5])
    pulse1.add_coherent_noise(sigmay(), 0, tlist_noise, coeff_noise)
    tlist_noise2 = np.array([0.5, 2, 3.])
    coeff_noise2 = np.array([0.1, 0.2, 0.3])
    pulse1.add_lindblad_noise(sigmax(), 1, coeff=True)
    pulse1.add_lindblad_noise(
        sigmax(), 0, tlist=tlist_noise2, coeff=coeff_noise2)

    assert_allclose(
        pulse1.get_ideal_qobjevo(2).ops[0].qobj, tensor(identity(2), sigmaz()))
    noise_qu, c_ops = pulse1.get_noisy_qobjevo(2)
    assert_allclose(noise_qu.tlist, np.array([0., 0.5,  1., 2., 2.5, 3.]))
    for ele in noise_qu.ops:
        if ele.qobj == tensor(identity(2), sigmaz()):
            assert_allclose(
                ele.coeff, np.array([0.1, 0.1, 0.2, 0.3, 0.3, 0.4]))
        elif ele.qobj == tensor(sigmay(), identity(2)):
            assert_allclose(
                ele.coeff, np.array([0., 0., 0.5, 0.5, 0.1, 0.5]))
    for c_op in c_ops:
        if len(c_op.ops) == 0:
            assert_allclose(c_ops[0].cte, tensor(identity(2), sigmax()))
        else:
            assert_allclose(
                c_ops[1].ops[0].qobj, tensor(sigmax(), identity(2)))
            assert_allclose(
                c_ops[1].tlist, np.array([0., 0.5, 1., 2., 2.5, 3.]))
            assert_allclose(
                c_ops[1].ops[0].coeff, np.array([0., 0.1, 0.1, 0.2, 0.2, 0.3]))


def TestPulseConstructor():
    """
    Test for creating empty Pulse, Pulse with constant coefficients etc.
    """
    coeff = np.array([0.1, 0.2, 0.3, 0.4])
    tlist = np.array([0., 1., 2., 3.])
    ham = sigmaz()
    # Special ways of initializing pulse
    pulse2 = Pulse(sigmax(), 0, tlist, True)
    assert_allclose(pulse2.get_ideal_qobjevo(2).ops[0].qobj,
                    tensor(sigmax(), identity(2)))

    pulse3 = Pulse(sigmay(), 0)
    assert_allclose(pulse3.get_ideal_qobjevo(2).cte.norm(), 0.)

    pulse4 = Pulse(None, None)  # Dummy empty ham
    assert_allclose(pulse4.get_ideal_qobjevo(2).cte.norm(), 0.)

    tlist_noise = np.array([1., 2.5, 3.])
    coeff_noise = np.array([0.5, 0.1, 0.5])
    tlist_noise2 = np.array([0.5, 2, 3.])
    coeff_noise2 = np.array([0.1, 0.2, 0.3])
    # Pulse with different dims
    random_qobj = Qobj(np.random.random((3, 3)))
    pulse5 = Pulse(sigmaz(), 1, tlist, True)
    pulse5.add_coherent_noise(sigmay(), 1, tlist_noise, coeff_noise)
    pulse5.add_lindblad_noise(
        random_qobj, 0, tlist=tlist_noise2, coeff=coeff_noise2)
    qu, c_ops = pulse5.get_noisy_qobjevo(dims=[3, 2])
    assert_allclose(qu.ops[0].qobj, tensor([identity(3), sigmaz()]))
    assert_allclose(qu.ops[1].qobj, tensor([identity(3), sigmay()]))
    assert_allclose(c_ops[0].ops[0].qobj, tensor([random_qobj, identity(2)]))


def TestDrift():
    """
    Test for Drift
    """
    drift = Drift()
    assert_allclose(drift.get_ideal_qobjevo(2).cte.norm(), 0)
    drift.add_drift(sigmaz(), targets=1)
    assert_allclose(
        drift.get_ideal_qobjevo(dims=[3, 2]).cte, tensor(identity(3), sigmaz()))
