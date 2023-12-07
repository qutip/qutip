from numpy.testing import assert_, assert_equal, assert_string_equal
from qutip.qip.algorithms.qft import qft, qft_steps, qft_gate_sequence
from qutip.qip.operations.gates import gate_sequence_product


class TestQFT:
    """
    A test class for the QuTiP functions for QFT
    """

    def testQFTComparison(self):
        """
        qft: compare qft and product of qft steps
        """
        for N in range(1, 5):
            U1 = qft(N)
            U2 = gate_sequence_product(qft_steps(N))
            assert_((U1 - U2).norm() < 1e-12)

    def testQFTGateSequenceNoSwapping(self):
        """
        qft: Inspect key properties of gate sequences of length N,
        with swapping disabled.
        """
        for N in range(1, 6):
            circuit = qft_gate_sequence(N, swapping=False)
            assert_equal(circuit.N, N)

            totsize = N * (N + 1) / 2
            assert_equal(len(circuit.gates), totsize)

            snots = sum(g.name == "SNOT" for g in circuit.gates)
            assert_equal(snots, N)

            phases = sum(g.name == "CPHASE" for g in circuit.gates)
            assert_equal(phases, N * (N - 1) / 2)

    def testQFTGateSequenceWithSwapping(self):
        """
        qft: Inspect swap gates added to gate sequences if
        swapping is enabled.
        """
        for N in range(1, 6):
            circuit = qft_gate_sequence(N, swapping=True)

            phases = int(N * (N + 1) / 2)
            swaps = int(N // 2)
            assert_equal(len(circuit.gates), phases + swaps)

            for i in range(phases, phases + swaps):
                assert_string_equal(circuit.gates[i].name, "SWAP")
