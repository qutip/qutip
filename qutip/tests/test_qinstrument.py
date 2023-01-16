from qutip import (
    QInstrument,
    Seq, Par,
    Pauli, PauliString
)

class TestSeqPar:
    """
    Unit tests that exercise the Seq/Par interface for instrument labels.
    """
    def test_seq_in_seq_is_flattened(self):
        s = Seq(Seq(1, 2), Seq(3, 4), 5)
        assert s == Seq(1, 2, 3, 4, 5)

    def test_par_in_par_is_flattened(self):
        s = Par(Par(1, 2), Par(3, 4), 5)
        assert s == Par(1, 2, 3, 4, 5)

    def test_seq_in_par_is_implied(self):
        s = Par(1, 2, Seq(3,), 4)
        assert s == Par(1, 2, 3, 4)

    def test_par_in_seq_is_implied(self):
        s = Seq(1, 2, Par(3,), 4)
        assert s == Seq(1, 2, 3, 4)

    def test_seq_parses_simple_case(self):
        s = Seq.parse("1,2,3,4")
        assert s == Seq(1, 2, 3, 4)

    def test_seq_parses_noninteger_labels(self):
        s = Seq.parse("⊥,a,3") # Only "3" should get converted to an int!
        assert s == Seq("⊥", "a", 3)

    def test_seq_parses_par(self):
        s = Seq.parse("0,1,2;3,4")
        assert s == Seq(Par(Seq(0, 1, 2), Seq(3, 4)))

    def test_seq_parses_and_trims(self):
        s = Seq.parse("0, 1, 2 ; 3, 4")
        assert s == Seq.parse("0,1,2;3,4")

    def test_repr(self):
        raise NotImplementedError("test not yet written")


class TestPauliString:
    """
    Unit tests that exercise the PauliSTring class used for labeling Pauli
    measurements.
    """

    def test_init_takes_mod(self):
        p = PauliString(2, "XY")
        q = PauliString(6, "XY")
        assert p == q

    def test_init_assumes_default_phase(self):
        p = PauliString("IXYZ")
        q = PauliString(0, "IXYZ")
        assert p == q

    def test_as_qobj(self):
        raise NotImplementedError("test not yet written")

    def test_neg(self):
        p = PauliString(3, "ZZYZX")
        q = PauliString(1, "ZZYZX")
        assert p == -q
        assert -p == q

    def test_parse_without_sign(self):
        raise NotImplementedError("test not yet written")

    def test_parse_with_sign(self):
        raise NotImplementedError("test not yet written")


class TestQInstrument:
    """
    Unit tests that exercise the QInstrument type for representing quantum
    instruments.
    """

    def test_init_from_qobj(self):
        raise NotImplementedError("test not yet written")

    def test_init_from_dict_with_str_labels(self):
        raise NotImplementedError("test not yet written")

    def test_init_from_dict_with_seq_labels(self):
        raise NotImplementedError("test not yet written")

    def test_nonselective_process_is_channel(self):
        raise NotImplementedError("test not yet written")

    def test_lmul(self):
        raise NotImplementedError("test not yet written")

    def test_rmul(self):
        raise NotImplementedError("test not yet written")

    def test_pow(self):
        raise NotImplementedError("test not yet written")    

    def test_tensor(self):
        raise NotImplementedError("test not yet written")

    def test_propagate(self):
        raise NotImplementedError("test not yet written")

    def test_sample(self):
        raise NotImplementedError("test not yet written")

    def test_mul_eliminates_impossible_outcomes(self):
        raise NotImplementedError("test not yet written")

    def test_finite_visibility(self):
        raise NotImplementedError("test not yet written")

    def test_complete(self):
        raise NotImplementedError("test not yet written")

    def test_reindex(self):
        raise NotImplementedError("test not yet written")

    def test_conditional_instruments(self):
        raise NotImplementedError("test not yet written")

    def test_ptrace(self):
        raise NotImplementedError("test not yet written")
