import qutip


def test_qutip_about(capsys):
    qutip.about()
    outerr = capsys.readouterr()
    out = outerr.out.splitlines()
    assert out[:3] == [
        "",
        "QuTiP: Quantum Toolbox in Python",
        "================================",
    ]
    assert out[-3:] == [
        "Please cite QuTiP in your publication.",
        "================================================================================",
        "For your convenience a bibtex reference can be easily generated using `qutip.cite()`",
    ]
    assert outerr.err == ""
