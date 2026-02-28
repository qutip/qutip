import pytest
import qutip
import numpy as np
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


def test_in_solver():
    opt = {"method": "adams", "store_states": True, "atol": 1}
    solver = qutip.SESolver(qutip.qeye(1), options=opt)
    adams = qutip.integrator.IntegratorScipyAdams
    lsoda = qutip.integrator.IntegratorScipylsoda
    bdf = qutip.integrator.IntegratorScipyBDF
    assert solver.options["store_states"] is True
    assert solver.options["method"] == "adams"
    assert solver.options["atol"] == 1
    assert solver.options["order"] == adams.integrator_options["order"]

    solver.options["method"] = "bdf"
    assert solver.options["store_states"] is True
    assert solver.options["method"] == "bdf"
    assert solver.options["atol"] == bdf.integrator_options["atol"]
    assert solver.options["order"] == bdf.integrator_options["order"]

    solver.options = {
        "method": "vern7",
        "store_final_state": True,
        "atol": 0.01
    }
    assert solver.options["store_states"] is True
    assert solver.options["store_final_state"] is True
    assert solver.options["method"] == "vern7"
    assert solver.options["atol"] == 0.01
    assert "order" not in solver.options
    assert "interpolate" in solver.options


def test_options_update_solver():
    opt = {"method": "adams", "normalize_output": False}
    solver = qutip.SESolver(1j * qutip.qeye(1), options=opt)

    solver.start(qutip.basis(1), 0)
    err_atol_def = (solver.step(1) - np.exp(1)).norm()

    solver.options["atol"] = 1
    solver.start(qutip.basis(1), 0)
    assert (solver.step(1) - np.exp(1)).norm() > err_atol_def * 10

    del solver.options["atol"]
    solver.start(qutip.basis(1), 0)
    assert (solver.step(1) - np.exp(1)).norm() == pytest.approx(err_atol_def)

    solver.options["atol"] = 1
    solver.options["atol"] = None
    solver.start(qutip.basis(1), 0)
    assert (solver.step(1) - np.exp(1)).norm() == pytest.approx(err_atol_def)

    solver.options["method"] = "diag"
    solver.start(qutip.basis(1), 0)
    assert (solver.step(1) - np.exp(1)).norm() == pytest.approx(0., abs=-1e10)
