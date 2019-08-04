from numpy.testing import assert_, run_module_suite, assert_allclose
import numpy as np

from qutip.qip.models.circuitprocessor import CircuitProcessor
from qutip.qip.models.circuitnoise import (
    RelaxationNoise, DecoherenceNoise, ControlAmpNoise, WhiteNoise, UserNoise)
from qutip.operators import qeye, sigmaz, sigmax, sigmay, destroy, identity
from qutip.tensor import tensor
from qutip.qobjevo import QobjEvo
from qutip.states import basis
from qutip.metrics import fidelity
from qutip.tensor import tensor

class DriftNoise(UserNoise):
    def __init__(self, op):
        self.op = op

    def get_noise(self, N, proc_qobjevo, dims=None):
        return QobjEvo(self.op), []


class TestCircuitNoise:
    def TestDecoherenceNoise(self):
        """
        Test for the decoherence noise
        """
        tlist = np.array([1, 2, 3, 4, 5, 6])
        coeffs = [np.array([1, 1, 1, 1, 1, 1])]

        # Time-dependent
        decnoise = DecoherenceNoise(
            sigmaz(), coeffs=coeffs, tlist=tlist, targets=[1])
        noise_list = decnoise.get_noise(2)
        assert_allclose(noise_list[0].ops[0].qobj, tensor(qeye(2), sigmaz()))
        assert_allclose(noise_list[0].ops[0].coeff, coeffs[0])
        assert_allclose(noise_list[0].tlist, tlist)

        # Time-indenpendent and all qubits
        decnoise = DecoherenceNoise(sigmax(), all_qubits=True)
        noise_list = decnoise.get_noise(2)
        assert_(tensor([qeye(2), sigmax()]) in noise_list)
        assert_(tensor([sigmax(), qeye(2)]) in noise_list)

        # Time-denpendent and all qubits
        decnoise = DecoherenceNoise(
            sigmax(), all_qubits=True, coeffs=coeffs*2, tlist=tlist)
        noise_list = decnoise.get_noise(2)
        assert_allclose(noise_list[0].ops[0].qobj, tensor(sigmax(), qeye(2)))
        assert_allclose(noise_list[0].ops[0].coeff, coeffs[0])
        assert_allclose(noise_list[0].tlist, tlist)
        assert_allclose(noise_list[1].ops[0].qobj, tensor(qeye(2), sigmax()))

    def TestRelaxationNoise(self):
        """
        Test for the relaxation noise
        """
        a = destroy(2)
        relnoise = RelaxationNoise(T1=[1., 1., 1.], T2=None)
        noise_list = relnoise.get_noise(3)
        assert_(len(noise_list) == 3)
        assert_allclose(noise_list[1], tensor([qeye(2), a, qeye(2)]))

        relnoise = RelaxationNoise(T1=None, T2=None)
        noise_list = relnoise.get_noise(2)
        assert_(len(noise_list) == 0)

        relnoise = RelaxationNoise(T1=None, T2=[0.2, 0.7])
        noise_list = relnoise.get_noise(2)
        assert_(len(noise_list) == 2)

        relnoise = RelaxationNoise(T1=[1., 1.], T2=[0.5, 0.5])
        noise_list = relnoise.get_noise(2)
        assert_(len(noise_list) == 4)

    def TestControlAmpNoise(self):
        """
        Test for the control amplitude noise
        """
        tlist = np.array([1, 2, 3, 4, 5, 6])
        coeff = np.array([1, 1, 1, 1, 1, 1])

        # no expand
        dummy_qobjevo = QobjEvo(sigmaz(), tlist=tlist)
        connoise = ControlAmpNoise(ops=sigmax(), coeffs=[coeff])
        noise = connoise.get_noise(N=1, proc_qobjevo=dummy_qobjevo)
        assert_allclose(noise.ops[0].qobj, sigmax())
        assert_allclose(noise.tlist, tlist)
        assert_allclose(noise.ops[0].coeff, coeff)

        dummy_qobjevo = QobjEvo(tensor([sigmaz(), sigmaz()]), tlist=tlist)
        connoise = ControlAmpNoise(ops=sigmay(), coeffs=[coeff], targets=[1])
        noise = connoise.get_noise(N=2, proc_qobjevo=dummy_qobjevo)
        assert_allclose(noise.ops[0].qobj, tensor([qeye(2), sigmay()]))

        # With expand
        dummy_qobjevo = QobjEvo(sigmaz(),tlist=tlist)
        connoise = ControlAmpNoise(
            ops=sigmaz(), coeffs=[coeff]*2, cyclic_permutation=True)
        noise = connoise.get_noise(N=2, proc_qobjevo=dummy_qobjevo)
        assert_allclose(noise.ops[0].qobj, tensor([sigmaz(), qeye(2)]))
        assert_allclose(noise.ops[1].qobj, tensor([qeye(2), sigmaz()]))

    def TestWhiteNoise(self):
        """
        Test for the white noise
        """

        tlist = np.array([1, 2, 3, 4, 5, 6])
        dummy_qobjevo = QobjEvo(sigmaz(), tlist=tlist)
        mean = 0.
        std = 0.5
        ops = [sigmaz(), sigmax()]
        whitenoise = WhiteNoise(mean=mean, std=std, ops=ops)
        noise = whitenoise.get_noise(N=1, proc_qobjevo=dummy_qobjevo)
        assert_allclose(noise.ops[1].qobj, sigmax())
        assert_allclose(len(noise.ops[1].coeff), len(tlist))
        assert_allclose(len(noise.ops), len(ops))

        # White noise with operator from processor (proc_qobjevo)
        coeff = np.array([1, 1, 1, 1, 1, 1])
        evo_ops = QobjEvo(
            [[sigmaz(), coeff], [sigmax(), coeff], [sigmay(), coeff]],
            tlist=tlist)
        whitenoise = WhiteNoise(mean=mean, std=std)
        noise = whitenoise.get_noise(N=1, proc_qobjevo=evo_ops)
        assert_allclose(noise.ops[1].qobj, sigmax())
        assert_(len(noise.ops[1].coeff) == len(tlist))
        assert_(len(noise.ops) == len(evo_ops.ops))

    def TestUserNoise(self):
        """
        Test for the user-defined noise object
        """
        dr_noise = DriftNoise(sigmax())
        proc = CircuitProcessor(1)
        proc.add_noise(dr_noise)
        proc.tlist = np.array([0, np.pi/2.])
        result = proc.run_state(rho0=basis(2, 0))
        assert_allclose(
            fidelity(result.states[-1], basis(2, 1)), 1, rtol=1.0e-6)


if __name__ == "__main__":
    run_module_suite()
