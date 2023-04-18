import pytest
import itertools
import qutip
from qutip.core.data.dispatch import Dispatcher
from qutip.core.data.base import SameData
import qutip.core.data as _data



class pseudo_dipatched:
    def __init__(self, types, output):
        self.use_samedata = SameData in types
        if output:
            self.output = types[-1]
            self.inputs = types[:-1]
        else:
            self.output = False
            self.inputs = types

    def __call__(self, *args, **kwargs):
        print(args)
        assert len(args) == len(self.inputs)
        assert not kwargs
        samedata = None
        for got, expected in zip(args, self.inputs):
            if expected is SameData and samedata is None:
                samedata = type(got)
                expected = type(got)
            elif expected is SameData:
                expected = samedata
            print(expected)
            assert isinstance(got, expected)

        if not self.output:
            return
        elif self.output is SameData:
            return _data.zeros[samedata](1,1)
        elif self.output is _data.Data:
            return _data.zeros(1,1)
        else:
            return _data.zeros[self.output](1,1)


def _gen_dispatcher(specialisations, output=False):
    def f(a=None, b=None, c=None, /):
        """
        Doc
        """

    n_inputs = len(specialisations[0]) - output
    dispatched = Dispatcher(f, ("a", "b", "c")[:n_inputs], output)
    dispatched.add_specialisations(
        [spec + (pseudo_dipatched(spec, output),) for spec in specialisations]
    )
    return dispatched


def _assert_usable_all_mix(dispatched, n_input, output):
    """
    Call the dispatched function with all mixes of data types, with and without
    output specification.
    """
    for in_types in itertools.product(_data.to.dtypes, repeat=n_input):
        ins = tuple(_data.zeros[dtype](1, 1) for dtype in in_types)

        if not output:
            out = dispatched(*ins)
            out = dispatched[in_types](*ins)
            continue

        out = dispatched(*ins)
        assert out is not None

        out = dispatched[in_types](*ins)
        assert out is not None

        if output:
            for out_dtype in _data.to.dtypes:

                out = dispatched[in_types + (out_dtype,)](*ins)
                if output:
                    assert isinstance(out, out_dtype)

                out = dispatched(*ins, dtype=out_dtype)
                if output:
                    assert isinstance(out, out_dtype)


def _test_name(arg):
    if isinstance(arg, bool):
        return str(arg)
    return ", ".join(spec.__name__ for spec in arg)


@pytest.mark.parametrize(['specialisation', "output"], [
    ((_data.Dense,), True),
    ((_data.Data,), True),
    ((_data.Dense,), False),
    ((_data.Data,), False),
    ((_data.Dense, _data.Dense), True),
    ((_data.Dense, _data.CSR), True),
    ((_data.Data, _data.Data), True),
    ((_data.Data, _data.Dense), True),
    ((_data.Dense, _data.Data), True),
    ((SameData, SameData), True),
    ((_data.Dense, _data.Dense), False),
    ((_data.Dense, _data.CSR), False),
    ((_data.Data, _data.Data), False),
    ((_data.Data, _data.Dense), False),
    ((_data.Dense, _data.Data), False),
    ((SameData, SameData), False),
    ((_data.Dense, _data.Dense, _data.Dense), True),
    ((_data.Dense, _data.CSR, _data.CSR), True),
    ((_data.Data, _data.Data, _data.Data), True),
    ((_data.Data, _data.Dense, _data.Dense), True),
    ((_data.Dense, _data.Data, _data.Data), True),
    ((SameData, SameData, SameData), True),
    ((SameData, SameData, _data.Dense), True),
    ((SameData, SameData, _data.Data), True),
    ((SameData, _data.Dense, SameData), True),
    ((SameData, _data.Data, SameData), True),
], ids=_test_name)
def test_build_full(specialisation, output):
    """
    Test that the dispatched function can always be called with any input type.
    """
    dispatched = _gen_dispatcher([specialisation], output)
    _assert_usable_all_mix(dispatched, len(specialisation) - output, output)
