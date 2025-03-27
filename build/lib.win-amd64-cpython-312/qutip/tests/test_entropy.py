import pytest
import numpy as np
import qutip


class TestVonNeumannEntropy:
    @pytest.mark.parametrize("p", np.linspace(0, 1, 17))
    def test_binary(self, p):
        dm = qutip.qdiags([p, 1 - p], 0)
        expected = 0 if p in [0, 1] else p*np.log2(p) + (1-p)*np.log2(1-p)
        assert abs(-qutip.entropy_vn(dm, 2) - expected) < 1e-12

    @pytest.mark.repeat(10)
    def test_pure_state(self):
        assert abs(qutip.entropy_vn(qutip.rand_ket(10))) < 1e-12


class TestLinearEntropy:
    @pytest.mark.repeat(10)
    def test_less_than_von_neumann(self):
        dm = qutip.rand_dm(10)
        assert qutip.entropy_linear(dm) <= qutip.entropy_vn(dm)

    @pytest.mark.repeat(10)
    def test_pure_state(self):
        assert abs(qutip.entropy_linear(qutip.rand_ket(10))) < 1e-12


class TestConcurrence:
    @pytest.mark.parametrize("dm", [
        pytest.param(qutip.bell_state(x).proj(), id='bell'+x)
        for x in ['00', '01', '10', '11']
    ])
    def test_maximally_entangled(self, dm):
        assert abs(qutip.concurrence(dm) - 1) < 1e-12

    @pytest.mark.repeat(10)
    def test_nonzero(self):
        dm = qutip.rand_dm([2, 2])
        assert qutip.concurrence(dm) >= 0


@pytest.mark.repeat(10)
class TestMutualInformation:
    def test_pure_state_additive(self):
        # Verify mutual information = S(A) + S(B) for pure states.
        dm = qutip.rand_dm([5, 5], distribution="pure")
        expect = (qutip.entropy_vn(dm.ptrace(0))
                  + qutip.entropy_vn(dm.ptrace(1)))
        assert abs(qutip.entropy_mutual(dm, [0], [1]) - expect) < 1e-13

    def test_component_selection(self):
        dm = qutip.rand_dm([2, 2, 2], distribution="pure")
        expect = (qutip.entropy_vn(dm.ptrace([0, 2]))
                  + qutip.entropy_vn(dm.ptrace(1)))
        assert abs(qutip.entropy_mutual(dm, [0, 2], [1]) - expect) < 1e-13


class TestRelativeEntropy:
    def _simple_relative_entropy_implementation(
            self, rho, sigma, log_base=np.log, tol=1e-12):
        """ A simplified relative entropy implementation for use in
            double-checking the optimised implementation within
            QuTiP itself.
        """
        # S(rho || sigma) = sum_i(p_i log p_i) - sum_ij(p_i P_ij log q_i)
        rvals, rvecs = rho.eigenstates()
        svals, svecs = sigma.eigenstates()
        rvecs = np.hstack([vec.full() for vec in rvecs]).T
        svecs = np.hstack([vec.full() for vec in svecs]).T
        # Calculate S
        S = 0
        for i in range(len(rvals)):
            if abs(rvals[i]) >= tol:
                S += rvals[i] * log_base(rvals[i])
            for j in range(len(svals)):
                P_ij = (
                    np.dot(rvecs[i], svecs[j].conjugate()) *
                    np.dot(svecs[j], rvecs[i].conjugate())
                )
                if abs(svals[j]) < tol and not (
                        abs(rvals[i]) < tol or abs(P_ij) < tol):
                    # kernel of sigma intersects support of rho
                    return np.inf
                if abs(svals[j]) >= tol:
                    S -= rvals[i] * P_ij * log_base(svals[j])
        return np.real(S)

    def test_rho_or_sigma_not_oper(self):
        rho = qutip.bra("00")
        sigma = qutip.bra("01")
        with pytest.raises(TypeError) as exc:
            qutip.entropy_relative(rho.dag(), sigma)
        assert str(exc.value) == "Inputs must be density matrices."
        with pytest.raises(TypeError) as exc:
            qutip.entropy_relative(rho, sigma.dag())
        assert str(exc.value) == "Inputs must be density matrices."
        with pytest.raises(TypeError) as exc:
            qutip.entropy_relative(rho, sigma)
        assert str(exc.value) == "Inputs must be density matrices."

    def test_rho_and_sigma_have_different_shape_and_dims(self):
        # test different shape and dims
        rho = qutip.ket("00")
        sigma = qutip.ket("0")
        with pytest.raises(ValueError) as exc:
            qutip.entropy_relative(rho, sigma)
        assert str(exc.value) == "Inputs must have the same shape and dims."
        # test same shape, difference dims
        rho = qutip.basis([2, 3], [0, 0])
        sigma = qutip.basis([3, 2], [0, 0])
        with pytest.raises(ValueError) as exc:
            qutip.entropy_relative(rho, sigma)
        assert str(exc.value) == "Inputs must have the same shape and dims."

    def test_base_not_2_or_e(self):
        rho = qutip.ket("00")
        sigma = qutip.ket("01")
        with pytest.raises(ValueError) as exc:
            qutip.entropy_relative(rho, sigma, base=3)
        assert str(exc.value) == "Base must be 2 or e."

    def test_infinite_relative_entropy(self):
        rho = qutip.ket("00")
        sigma = qutip.ket("01")
        assert qutip.entropy_relative(rho, sigma) == np.inf

    def test_base_2_or_e(self):
        rho = qutip.ket2dm(qutip.ket("00"))
        sigma = rho + qutip.ket2dm(qutip.ket("01"))
        sigma = sigma.unit()
        assert (
            qutip.entropy_relative(rho, sigma) == pytest.approx(np.log(2))
        )
        assert (
            qutip.entropy_relative(rho, sigma, base=np.e)
            == pytest.approx(0.69314718)
        )
        assert qutip.entropy_relative(rho, sigma, base=2) == pytest.approx(1)

    def test_pure_vs_maximally_mixed_state(self):
        rho = qutip.ket("00")
        sigma = sum(
            qutip.ket2dm(qutip.ket(psi)) for psi in ["00", "01", "10", "11"]
        ).unit()
        assert qutip.entropy_relative(rho, sigma, base=2) == pytest.approx(2)

    def test_density_matrices_with_non_real_eigenvalues(self):
        rho = qutip.ket2dm(qutip.ket("00"))
        sigma = qutip.ket2dm(qutip.ket("01"))
        with pytest.raises(ValueError) as exc:
            qutip.entropy_relative(rho + 1j, sigma)
        assert str(exc.value) == "Input rho has non-real eigenvalues."
        with pytest.raises(ValueError) as exc:
            qutip.entropy_relative(rho - 1j, sigma)
        assert str(exc.value) == "Input rho has non-real eigenvalues."
        with pytest.raises(ValueError) as exc:
            qutip.entropy_relative(rho, sigma + 1j)
        assert str(exc.value) == "Input sigma has non-real eigenvalues."
        with pytest.raises(ValueError) as exc:
            qutip.entropy_relative(rho, sigma - 1j)
        assert str(exc.value) == "Input sigma has non-real eigenvalues."

    @pytest.mark.repeat(20)
    def test_random_dm_with_self(self):
        rho = qutip.rand_dm(8)
        rel = qutip.entropy_relative(rho, rho)
        assert abs(rel) < 1e-13

    @pytest.mark.repeat(20)
    def test_random_rho_sigma(self):
        rho = qutip.rand_dm(8)
        sigma = qutip.rand_dm(8)
        rel = qutip.entropy_relative(rho, sigma)
        assert rel >= 0
        assert rel == pytest.approx(
            self._simple_relative_entropy_implementation(rho, sigma, np.log)
        )


@pytest.mark.repeat(20)
class TestConditionalEntropy:
    def test_inequality_3_qubits(self):
        # S(A | B,C) <= S(A|B)
        full = qutip.rand_dm([2]*3, distribution="pure")
        ab = full.ptrace([0, 1])
        assert (qutip.entropy_conditional(full, [1, 2])
                <= qutip.entropy_conditional(ab, 1))

    def test_triangle_inequality_4_qubits(self):
        # S(A,B | C,D) <= S(A|C) + S(B|D)
        full = qutip.rand_dm([2]*4, distribution="pure")
        ac, bd = full.ptrace([0, 2]), full.ptrace([1, 3])
        assert (qutip.entropy_conditional(full, [2, 3])
                <= (qutip.entropy_conditional(ac, 1)
                    + qutip.entropy_conditional(bd, 1)))


_alpha = 2*np.pi * np.random.rand()


@pytest.mark.parametrize(["gate", "expected"], [
    pytest.param(qutip.gates.cnot(), 2/9, id="CNOT"),
    pytest.param(qutip.gates.iswap(), 2/9, id="ISWAP"),
    pytest.param(qutip.gates.berkeley(), 2/9, id="Berkeley"),
    pytest.param(qutip.gates.swap(), 0, id="SWAP"),
    pytest.param(qutip.gates.sqrtswap(), 1/6, id="sqrt(SWAP)"),
    pytest.param(qutip.gates.swapalpha(_alpha),
                 np.sin(np.pi*_alpha)**2 / 6, id="SWAP(alpha)"),
])
def test_entangling_power(gate, expected):
    assert abs(qutip.entangling_power(gate) - expected) < 1e-12
