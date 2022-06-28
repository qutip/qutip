import pytest
import qutip
from qutip.solver.options import SolverOptions


@pytest.fixture(params=['sesolve', 'mesolve', 'brmesolve'])
def solver(request):
    return request.param


def test_SolverOptions_Error():
    opt = SolverOptions("sesolve")
    assert not opt     # Empty solver is false
    assert not {**opt} # Empty solver convect to empty dict
    with pytest.raises(KeyError) as error:
        SolverOptions("sesolve", bad_options=5)
    assert "bad_options" in str(error.value)

    with pytest.raises(ValueError) as error:
        SolverOptions("Not a solver")
    assert "Not a solver" in str(error.value)


def test_empty_SolverOptions():
    opt = SolverOptions(tensor_type=1, atol=1, dummy=True)
    assert opt["tensor_type"] == 1
    assert opt["atol"] == 1
    assert opt["dummy"] is True
    assert "dummy" in opt
    assert "dummy" not in opt.convert('brmesolve')
    assert opt.convert('brmesolve')["tensor_type"] == 1


def _assert_keys(keys, **kwargs):
    for key in keys:
        assert key in kwargs
    assert not (keys ^ kwargs.keys())


def test_SolverOptions_dictlike(solver):
    opt_dict = {
        "method": "bdf",
        "atol": 1,
        "store_states": True,
    }
    opt = SolverOptions(solver, **opt_dict)
    assert opt
    assert opt["method"] == "bdf"
    assert opt["atol"] == 1
    assert opt["store_states"] is True

    _assert_keys(opt_dict.keys(), **opt)
    assert opt_dict.keys() == opt.keys()
    for key in opt_dict:
        assert key in opt
    assert "rtol" in opt

    copy = opt.copy()
    assert opt.items() == copy.items()
    copy['rtol'] = 2
    assert opt.items() != copy.items()

    for val in opt.values():
        assert val in opt_dict.values()
    assert len(opt.values()) == 3

    for key, val in opt.items():
        assert val == opt_dict[key]
    assert len(opt.items()) == 3

    del opt["atol"]
    assert len(opt.items()) == 2
    assert "atol" not in opt.items()


def test_SolverOptions_Feedback(solver):
    called = []

    def _catch(keys):
        called.append(keys.pop())

    opt = SolverOptions(solver, _solver_feedback=_catch)
    opt["method"] = "lsoda"
    opt["atol"] = 1e-5

    assert "method" in called
    assert "atol" in called
    assert len(called) == 2


def test_SolverOptions_convert():
    opt_br = SolverOptions("brmesolve", sparse_eigensolver=True, atol=1)
    opt_se = opt_br.convert("sesolve")

    assert opt_br.solver == "brmesolve"
    assert "sparse_eigensolver" in opt_br
    assert opt_br["atol"] == 1

    assert opt_se.solver == "sesolve"
    assert "sparse_eigensolver" not in opt_se
    assert opt_se["atol"] == 1
