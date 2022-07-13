import pytest
import qutip
from qutip.solver.options import _SolverOptions


default = {
    "opt1": False,
    "opt2": True,
    "opt3": None,
}


def test_default():
    opt = _SolverOptions(default)
    assert opt
    assert default.keys() == opt.keys()
    assert opt["opt1"] is False
    assert opt["opt2"] is True
    assert opt["opt3"] is None


def test_SolverOptions_dictlike():
    opt_dict = {
        "opt1": True,
        "opt2": None,
    }
    opt = _SolverOptions(default, **opt_dict)
    assert opt
    assert default.keys() == opt.keys()
    assert opt["opt1"] is True
    assert opt["opt2"] is None
    assert opt["opt3"] is None

    copy = opt.copy()
    assert opt.items() == copy.items()
    copy['opt3'] = 2
    assert opt.items() != copy.items()

    assert len(opt.values()) == 3
    assert len(opt.items()) == 3
    assert len(opt.keys()) == 3


def test_del():
    opt_dict = {
        "opt1": True,
    }
    opt = _SolverOptions(default, opt1=True)
    del opt["opt1"]
    assert opt["opt1"] is False


def test_SolverOptions_Feedback():
    called = []

    def _catch(keys):
        assert isinstance(keys, (set, str))
        called.append(keys)

    opt = _SolverOptions(default, _catch)
    opt["opt1"] = 2
    opt["opt2"] = 1e-5

    assert "opt1" in called
    assert "opt2" in called
    assert len(called) == 2


def test_error():
    with pytest.raises(KeyError) as err:
        _SolverOptions(default, opt4=4)
    assert "opt4" in str(err.value)

    with pytest.raises(KeyError) as err:
        opt = _SolverOptions(default)
        opt["opt4"] = 4
    assert "opt4" in str(err.value)


def test_print():
    opt = _SolverOptions(default, None, "Test options", "Custom doc")
    assert "Test options" in opt.__str__()
    assert "opt1 : False" in opt.__str__()
    assert "opt2 : True" in opt.__str__()
    assert "opt3 : None" in opt.__str__()
    assert opt.__doc__ == "Custom doc"
