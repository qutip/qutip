# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

import numpy as np
from numpy.testing import assert_, run_module_suite, assert_allclose

from qutip import (
    Qobj, sigmax, sigmay, sigmaz, identity, tensor, QobjEvo
)
from qutip.qip.pulse import Pulse, Drift
from qutip import coefficient


def _compare_qobjevo(qevo1, qevo2, t_min, t_max):
    return all([np.allclose(qevo1(t).full(), qevo2(t).full())
                for t in t_min + np.random.rand(25) * (t_max - t_min)])


def test_BasicPulse():
    """
    Test for basic pulse generation and attributes.
    """
    coeff = np.array([0.1, 0.2, 0.3, 0.4])
    tlist = np.array([0., 1., 2., 3.])
    ham = sigmaz()

    # Basic tests
    pulse1 = Pulse(ham, 1, tlist, coeff)
    assert_allclose(
        pulse1.get_ideal_qobjevo(2)(0).full(),
        tensor(identity(2), sigmaz()).full() * 0.1)
    pulse1.tlist = 2 * tlist
    assert_allclose(pulse1.tlist, 2 * tlist)
    pulse1.tlist = tlist
    pulse1.coeff = 2 * coeff
    assert_allclose(pulse1.coeff, 2 * coeff)
    pulse1.coeff = coeff
    pulse1.qobj = 2 * sigmay()
    assert_allclose(pulse1.qobj.full(), 2 * sigmay().full())
    pulse1.qobj = ham
    pulse1.targets = 3
    assert_allclose(pulse1.targets, 3)
    pulse1.targets = 1
    assert_allclose(pulse1.get_ideal_qobj(2).full(),
                    tensor(identity(2), sigmaz()).full())


def test_CoherentNoise():
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
        pulse1.get_ideal_qobjevo(2)(0).full(),
        tensor(identity(2), sigmaz()).full() * 0.1)
    assert_(len(pulse1.coherent_noise) == 1)
    noise_qu, c_ops = pulse1.get_noisy_qobjevo(2)
    assert_allclose(c_ops, [])
    assert_allclose(pulse1.get_full_tlist(), np.array([0., 1., 2., 3.]))

    expected = QobjEvo([
        [tensor(identity(2), sigmaz()), np.array([0.1, 0.2, 0.3, 0.4])],
        [tensor(sigmay(), identity(2)), np.array([0.1, 0.2, 0.3, 0.4])]
    ], tlist = np.array([0., 1., 2., 3.]))
    assert _compare_qobjevo(noise_qu, expected, 0, 3)


def test_NoisyPulse():
    """
    Test for lindblad noise and different tlist
    """
    coeff = np.array([0.1, 0.2, 0.3, 0.4])
    tlist = np.array([0., 1., 2., 3.])
    ham = sigmaz()
    pulse1 = Pulse(ham, 1, tlist, coeff)
    # Add coherent noise and lindblad noise with different tlist
    pulse1.spline_kind = "step_func"
    tlist_noise = np.array([0., 1., 2.5, 3.])
    coeff_noise = np.array([0., 0.5, 0.1, 0.5])
    pulse1.add_coherent_noise(sigmay(), 0, tlist_noise, coeff_noise)
    tlist_noise2 = np.array([0., 0.5, 2, 3.])
    coeff_noise2 = np.array([0., 0.1, 0.2, 0.3])
    pulse1.add_lindblad_noise(sigmax(), 1, coeff=True)
    pulse1.add_lindblad_noise(
        sigmax(), 0, tlist=tlist_noise2, coeff=coeff_noise2)

    assert_allclose(
        pulse1.get_ideal_qobjevo(2)(0).full(),
        tensor(identity(2), sigmaz()).full() * 0.1)
    noise_qu, c_ops = pulse1.get_noisy_qobjevo(2)
    assert_allclose(pulse1.get_full_tlist(), np.array([0., 0.5,  1., 2., 2.5, 3.]))
    expected = QobjEvo([
        [tensor(identity(2), sigmaz()), np.array([0.1, 0.1, 0.2, 0.3, 0.3, 0.4])],
        [tensor(sigmay(), identity(2)), np.array([0., 0., 0.5, 0.5, 0.1, 0.5])]
    ], tlist=np.array([0., 0.5,  1., 2., 2.5, 3.]), args={"_step_func_coeff":True})
    assert _compare_qobjevo(noise_qu, expected, 0, 3)

    for c_op in c_ops:
        if c_op.isconstant:
            assert_allclose(c_op(0).full(),
                            tensor(identity(2), sigmax()).full())
        else:
            expected = QobjEvo([tensor(sigmax(), identity(2)),
                                np.array([0., 0.1, 0.1, 0.2, 0.2, 0.3])],
                               tlist=np.array([0., 0.5,  1., 2., 2.5, 3.]),
                               args={"_step_func_coeff":True})
            assert _compare_qobjevo(c_op, expected, 0, 3)


def test_PulseConstructor():
    """
    Test for creating empty Pulse, Pulse with constant coefficients etc.
    """
    coeff = np.array([0.1, 0.2, 0.3, 0.4])
    tlist = np.array([0., 1., 2., 3.])
    ham = sigmaz()
    # Special ways of initializing pulse
    pulse2 = Pulse(sigmax(), 0, tlist, True)
    assert_allclose(pulse2.get_ideal_qobjevo(2)(0).full(),
                    tensor(sigmax(), identity(2)).full())

    pulse3 = Pulse(sigmay(), 0)
    assert_allclose(pulse3.get_ideal_qobjevo(2)(0).norm(), 0.)

    pulse4 = Pulse(None, None)  # Dummy empty ham
    assert_allclose(pulse4.get_ideal_qobjevo(2)(0).norm(), 0.)

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
    expected = QobjEvo([tensor([identity(3), sigmaz()]),
        [tensor([identity(3), sigmay()]), coeff_noise]
    ], tlist=tlist_noise)
    assert _compare_qobjevo(qu, expected, 0, 3)
    assert_allclose(c_ops[0](0.5).full(),
                    tensor([random_qobj, identity(2)]).full() * 0.1)


def test_Drift():
    """
    Test for Drift
    """
    drift = Drift()
    assert_allclose(drift.get_ideal_qobjevo(2)(0).norm(), 0)
    drift.add_drift(sigmaz(), targets=1)
    assert_allclose(
        drift.get_ideal_qobjevo(dims=[3, 2])(0).full(),
        tensor(identity(3), sigmaz()).full())
