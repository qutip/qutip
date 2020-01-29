from numpy.testing import assert_, run_module_suite, assert_allclose
import numpy as np

from qutip.qip.device.processor import Processor
from qutip.qip.noise import (
    RelaxationNoise, DecoherenceNoise, ControlAmpNoise, RandomNoise, UserNoise)
from qutip.operators import qeye, sigmaz, sigmax, sigmay, destroy, identity
from qutip.tensor import tensor
from qutip.qobjevo import QobjEvo
from qutip.states import basis
from qutip.metrics import fidelity
from qutip.tensor import tensor
from qutip.qip.pulse import Pulse


class DriftNoise(UserNoise):
    def __init__(self, op):
        self.qobj = op

    def get_noisy_dynamics(self, ctrl_pulses, dims=None):
        dummy = Pulse(None, None)
        dummy.add_coherent_noise(self.qobj, 0, coeff=True)
        return ctrl_pulses + [dummy]


class TestNoise:
    def TestDecoherenceNoise(self):
        """
        Test for the decoherence noise
        """
        tlist = np.array([1, 2, 3, 4, 5, 6])
        coeff = np.array([1, 1, 1, 1, 1, 1])

        # Time-dependent
        decnoise = DecoherenceNoise(
            sigmaz(), coeff=coeff, tlist=tlist, targets=[1])
        noisy_qu, c_ops = decnoise.get_noisy_dynamics(2).get_noisy_qobjevo(dims=2)
        assert_allclose(c_ops[0].ops[0].qobj, tensor(qeye(2), sigmaz()))
        assert_allclose(c_ops[0].ops[0].coeff, coeff)
        assert_allclose(c_ops[0].tlist, tlist)

        # Time-indenpendent and all qubits
        decnoise = DecoherenceNoise(sigmax(), all_qubits=True)
        noisy_qu, c_ops = decnoise.get_noisy_dynamics(2).get_noisy_qobjevo(dims=2)
        c_ops = [qu.cte for qu in c_ops]
        assert_(tensor([qeye(2), sigmax()]) in c_ops)
        assert_(tensor([sigmax(), qeye(2)]) in c_ops)

        # Time-denpendent and all qubits
        decnoise = DecoherenceNoise(
            sigmax(), all_qubits=True, coeff=coeff*2, tlist=tlist)
        noisy_qu, c_ops = decnoise.get_noisy_dynamics(2).get_noisy_qobjevo(dims=2)
        assert_allclose(c_ops[0].ops[0].qobj, tensor(sigmax(), qeye(2)))
        assert_allclose(c_ops[0].ops[0].coeff, coeff*2)
        assert_allclose(c_ops[0].tlist, tlist)
        assert_allclose(c_ops[1].ops[0].qobj, tensor(qeye(2), sigmax()))

    def TestRelaxationNoise(self):
        """
        Test for the relaxation noise
        """
        a = destroy(2)
        relnoise = RelaxationNoise(t1=[1., 1., 1.], t2=None)
        noisy_qu, c_ops = relnoise.get_noisy_dynamics(3).get_noisy_qobjevo(dims=3)
        assert_(len(c_ops) == 3)
        assert_allclose(c_ops[1].cte, tensor([qeye(2), a, qeye(2)]))

        relnoise = RelaxationNoise(t1=None, t2=None)
        noisy_qu, c_ops = relnoise.get_noisy_dynamics(2).get_noisy_qobjevo(dims=2)
        assert_(len(c_ops) == 0)

        relnoise = RelaxationNoise(t1=None, t2=[0.2, 0.7])
        noisy_qu, c_ops = relnoise.get_noisy_dynamics(2).get_noisy_qobjevo(dims=2)
        assert_(len(c_ops) == 2)

        relnoise = RelaxationNoise(t1=[1., 1.], t2=[0.5, 0.5])
        noisy_qu, c_ops = relnoise.get_noisy_dynamics(2).get_noisy_qobjevo(dims=2)
        assert_(len(c_ops) == 4)

    def TestControlAmpNoise(self):
        """
        Test for the control amplitude noise
        """
        tlist = np.array([1, 2, 3, 4, 5, 6])
        coeff = np.array([1, 1, 1, 1, 1, 1])

        # use proc_qobjevo
        pulses = [Pulse(sigmaz(), 0, tlist, coeff)]
        connoise = ControlAmpNoise(coeff=coeff, tlist=tlist)
        noisy_pulses = connoise.get_noisy_dynamics(pulses=pulses)
        assert_allclose(noisy_pulses[0].coherent_noise[0].qobj, sigmaz())
        assert_allclose(noisy_pulses[0].coherent_noise[0].coeff, coeff)

    def TestRandomNoise(self):
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
        noisy_pulses = gaussnoise.get_noisy_dynamics(pulses=pulses)
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
        noisy_pulses = gaussnoise.get_noisy_dynamics(pulses=pulses)
        assert_allclose(
            noisy_pulses[0].coherent_noise[0].tlist,
            np.linspace(1, 6, int(5/0.2) + 1))
        assert_allclose(
            noisy_pulses[1].coherent_noise[0].tlist,
            np.linspace(1, 6, int(5/0.2) + 1))
        assert_allclose(
            noisy_pulses[2].coherent_noise[0].tlist,
            np.linspace(1, 6, int(5/0.2) + 1))

    def TestUserNoise(self):
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


if __name__ == "__main__":
    run_module_suite()
