from numpy.testing import assert_, assert_allclose
import numpy as np

from qutip.qip.device.processor import Processor
from qutip.qip.noise import (
    Noise, RelaxationNoise, DecoherenceNoise, ControlAmpNoise, RandomNoise)
from qutip.operators import qeye, sigmaz, sigmax, sigmay, destroy, identity
from qutip.tensor import tensor
from qutip.qobjevo import QobjEvo
from qutip.states import basis
from qutip.metrics import fidelity
from qutip.tensor import tensor
from qutip.qip.pulse import Pulse


class DriftNoise(Noise):
    def __init__(self, op):
        self.qobj = op

    def get_noisy_dynamics(self, dims, pulses, systematic_noise):
        systematic_noise.add_coherent_noise(self.qobj, 0, coeff=True)
        # test the backward compatibility
        return pulses


class TestNoise:
    def test_decoherence_noise(self):
        """
        Test for the decoherence noise
        """
        tlist = np.array([1, 2, 3, 4, 5, 6])
        coeff = np.array([1, 1, 1, 1, 1, 1])

        # Time-dependent
        decnoise = DecoherenceNoise(
            sigmaz(), coeff=coeff, tlist=tlist, targets=[1])
        dims = [2] * 2
        pulses, systematic_noise = decnoise.get_noisy_dynamics(dims=dims)
        noisy_qu, c_ops = systematic_noise.get_noisy_qobjevo(dims=dims)
        assert_allclose(c_ops[0].ops[0].qobj, tensor(qeye(2), sigmaz()))
        assert_allclose(c_ops[0].ops[0].coeff, coeff)
        assert_allclose(c_ops[0].tlist, tlist)

        # Time-indenpendent and all qubits
        decnoise = DecoherenceNoise(sigmax(), all_qubits=True)
        pulses, systematic_noise = decnoise.get_noisy_dynamics(dims=dims)
        noisy_qu, c_ops = systematic_noise.get_noisy_qobjevo(dims=dims)
        c_ops = [qu.cte for qu in c_ops]
        assert_(tensor([qeye(2), sigmax()]) in c_ops)
        assert_(tensor([sigmax(), qeye(2)]) in c_ops)

        # Time-denpendent and all qubits
        decnoise = DecoherenceNoise(
            sigmax(), all_qubits=True, coeff=coeff*2, tlist=tlist)
        pulses, systematic_noise = decnoise.get_noisy_dynamics(dims=dims)
        noisy_qu, c_ops = systematic_noise.get_noisy_qobjevo(dims=dims)
        assert_allclose(c_ops[0].ops[0].qobj, tensor(sigmax(), qeye(2)))
        assert_allclose(c_ops[0].ops[0].coeff, coeff*2)
        assert_allclose(c_ops[0].tlist, tlist)
        assert_allclose(c_ops[1].ops[0].qobj, tensor(qeye(2), sigmax()))

    def test_relaxation_noise(self):
        """
        Test for the relaxation noise
        """
        # only t1
        a = destroy(2)
        dims = [2] * 3
        relnoise = RelaxationNoise(t1=[1., 1., 1.], t2=None)
        systematic_noise = Pulse(None, None, label="system")
        pulses, systematic_noise = relnoise.get_noisy_dynamics(
            dims=dims, systematic_noise=systematic_noise)
        noisy_qu, c_ops = systematic_noise.get_noisy_qobjevo(dims=dims)
        assert_(len(c_ops) == 3)
        assert_allclose(c_ops[1].cte, tensor([qeye(2), a, qeye(2)]))

        # no relaxation
        dims = [2] * 2
        relnoise = RelaxationNoise(t1=None, t2=None)
        systematic_noise = Pulse(None, None, label="system")
        pulses, systematic_noise = relnoise.get_noisy_dynamics(
            dims=dims, systematic_noise=systematic_noise)
        noisy_qu, c_ops = systematic_noise.get_noisy_qobjevo(dims=dims)
        assert_(len(c_ops) == 0)

        # only t2
        relnoise = RelaxationNoise(t1=None, t2=[0.2, 0.7])
        systematic_noise = Pulse(None, None, label="system")
        pulses, systematic_noise = relnoise.get_noisy_dynamics(
            dims=dims, systematic_noise=systematic_noise)
        noisy_qu, c_ops = systematic_noise.get_noisy_qobjevo(dims=dims)
        assert_(len(c_ops) == 2)

        # t1+t2 and systematic_noise = None
        relnoise = RelaxationNoise(t1=[1., 1.], t2=[0.5, 0.5])
        pulses, systematic_noise = relnoise.get_noisy_dynamics(dims=dims)
        noisy_qu, c_ops = systematic_noise.get_noisy_qobjevo(dims=dims)
        assert_(len(c_ops) == 4)

    def test_control_amplitude_noise(self):
        """
        Test for the control amplitude noise
        """
        tlist = np.array([1, 2, 3, 4, 5, 6])
        coeff = np.array([1, 1, 1, 1, 1, 1])

        # use proc_qobjevo
        pulses = [Pulse(sigmaz(), 0, tlist, coeff)]
        connoise = ControlAmpNoise(coeff=coeff, tlist=tlist)
        noisy_pulses, systematic_noise = \
            connoise.get_noisy_dynamics(pulses=pulses)
        assert_allclose(pulses[0].coherent_noise[0].qobj, sigmaz())
        assert_allclose(noisy_pulses[0].coherent_noise[0].coeff, coeff)

    def test_random_noise(self):
        """
        Test for the white noise
        """
        tlist = np.array([1, 2, 3, 4, 5, 6])
        coeff = np.array([1, 1, 1, 1, 1, 1])
        dummy_qobjevo = QobjEvo(sigmaz(), tlist=tlist)
        mean = 0.
        std = 0.5
        pulses = [Pulse(sigmaz(), 0, tlist, coeff),
                  Pulse(sigmax(), 0, tlist, coeff*2),
                  Pulse(sigmay(), 0, tlist, coeff*3)]

        # random noise with operators from proc_qobjevo
        gaussnoise = RandomNoise(
            dt=0.1, rand_gen=np.random.normal, loc=mean, scale=std)
        noisy_pulses, systematic_noise = \
            gaussnoise.get_noisy_dynamics(pulses=pulses)
        assert_allclose(noisy_pulses[2].qobj, sigmay())
        assert_allclose(noisy_pulses[1].coherent_noise[0].qobj, sigmax())
        assert_allclose(
            len(noisy_pulses[0].coherent_noise[0].tlist),
            len(noisy_pulses[0].coherent_noise[0].coeff))

        # random noise with dt and other random number generator
        pulses = [Pulse(sigmaz(), 0, tlist, coeff),
                  Pulse(sigmax(), 0, tlist, coeff*2),
                  Pulse(sigmay(), 0, tlist, coeff*3)]
        gaussnoise = RandomNoise(lam=0.1, dt=0.2, rand_gen=np.random.poisson)
        assert_(gaussnoise.rand_gen is np.random.poisson)
        noisy_pulses, systematic_noise = \
            gaussnoise.get_noisy_dynamics(pulses=pulses)
        assert_allclose(
            noisy_pulses[0].coherent_noise[0].tlist,
            np.linspace(1, 6, int(5/0.2) + 1))
        assert_allclose(
            noisy_pulses[1].coherent_noise[0].tlist,
            np.linspace(1, 6, int(5/0.2) + 1))
        assert_allclose(
            noisy_pulses[2].coherent_noise[0].tlist,
            np.linspace(1, 6, int(5/0.2) + 1))

    def test_user_defined_noise(self):
        """
        Test for the user-defined noise object
        """
        dr_noise = DriftNoise(sigmax())
        proc = Processor(1)
        proc.add_noise(dr_noise)
        tlist = np.array([0, np.pi/2.])
        proc.add_pulse(Pulse(identity(2), 0, tlist, False))
        result = proc.run_state(init_state=basis(2, 0))
        assert_allclose(
            fidelity(result.states[-1], basis(2, 1)), 1, rtol=1.0e-6)
