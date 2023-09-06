import pytest
import itertools
import qutip
from qutip.core.data.dispatch import Dispatcher, _constructed_specialisation
import qutip.core.data as _data


class pseudo_dipatched:
    def __init__(self, types, output):
        if output:
            self.output = types[-1]
            self.inputs = types[:-1]
        else:
            self.output = False
            self.inputs = types

    def __call__(self, *args, **kwargs):
        assert len(args) == len(self.inputs)
        assert not kwargs
        for got, expected in zip(args, self.inputs):
            assert isinstance(got, expected)

        if not self.output:
            return
        elif self.output is _data.Data:
            return _data.zeros(1,1)
        else:
            return _data.zeros[self.output](1,1)


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
    ((_data.Dense, _data.Dense), False),
    ((_data.Dense, _data.CSR), False),
    ((_data.Data, _data.Data), False),
    ((_data.Data, _data.Dense), False),
    ((_data.Dense, _data.Data), False),
    ((_data.Dense, _data.Dense, _data.Dense), True),
    ((_data.Dense, _data.CSR, _data.CSR), True),
    ((_data.Data, _data.Data, _data.Data), True),
    ((_data.Data, _data.Dense, _data.Dense), True),
    ((_data.Dense, _data.Data, _data.Data), True),
], ids=_test_name)
def test_build_full(specialisation, output):
    """
    Test that the dispatched function can always be called with any input type.
    """
    def f(a=None, b=None, c=None, /):
        """
        Doc
        """


    n_input = len(specialisation) - output
    dispatched = Dispatcher(f, ("a", "b", "c")[:n_input], output)
    dispatched.add_specialisations(
        [specialisation + (pseudo_dipatched(specialisation, output),)]
    )

    for in_types in itertools.product(_data.to.dtypes, repeat=n_input):
        ins = tuple(_data.zeros[dtype](1, 1) for dtype in in_types)

        if not output:
            out = dispatched(*ins)
            out = dispatched[in_types](*ins)

        else:
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


def test_Data_low_priority_one_dispatch():
    class func():
        __name__ = "dummy name"
        def __call__(self, a, /):
            return _data.zeros[_data.Dense](1, 1)

    f_dense = func()
    f_data = func()

    dispatched = Dispatcher(f_dense, ("a",), False)
    dispatched.add_specialisations([
        (_data.Dense, f_dense), (_data.Data, f_data)]
    )

    assert dispatched[_data.Dense] is f_dense
    assert dispatched[_data.CSR] is f_data

    dispatched = Dispatcher(f_dense, (), True)
    dispatched.add_specialisations([
        (_data.Dense, f_dense), (_data.Data, f_data)]
    )

    assert dispatched[_data.Dense] is f_dense
    assert isinstance(dispatched[_data.CSR], _constructed_specialisation)


def test_Data_low_priority_two_dispatch():
    class func():
        __name__ = ""
        def __init__(self):
            self.count = 0

        def __call__(self, a=None, b=None, /):
            self.count += 1
            return _data.zeros[_data.Dense](1, 1)

    f_dense = func()
    f_mixed = func()
    f_data = func()

    dispatched = Dispatcher(f_dense, ("a", "b"), False)
    dispatched.add_specialisations([
        (_data.Dense, _data.Dense, f_dense),
        (_data.Dense, _data.Data, f_mixed),
        (_data.Data, _data.Data, f_data),
    ])

    assert dispatched[_data.Dense, _data.Dense] is f_dense
    assert dispatched[_data.Dense, _data.CSR] is f_mixed
    assert dispatched[_data.CSR, _data.Dense] is f_data
    assert dispatched[_data.CSR, _data.CSR] is f_data

    dispatched = Dispatcher(f_dense, ("a",), True)
    dispatched.add_specialisations([
        (_data.Dense, _data.Dense, f_dense),
        (_data.Dense, _data.Data, f_mixed),
        (_data.Data, _data.Data, f_data),
    ])

    assert dispatched[_data.Dense, _data.Dense] is f_dense
    assert dispatched[_data.CSR] is f_data
    assert dispatched[_data.Dense] is f_dense
    assert isinstance(
        dispatched[_data.Dense, _data.CSR], _constructed_specialisation
    )

    assert f_mixed.count == 0
    dispatched[_data.Dense, _data.CSR](_data.zeros[_data.Dense](1, 1))
    assert f_mixed.count == 1

    assert isinstance(
        dispatched[_data.CSR, _data.Dense], _constructed_specialisation
    )
    assert isinstance(
        dispatched[_data.CSR, _data.CSR], _constructed_specialisation
    )

    assert f_data.count == 0
    dispatched[_data.CSR, _data.Dense](_data.zeros[_data.CSR](1, 1))
    dispatched[_data.CSR, _data.CSR](_data.zeros[_data.CSR](1, 1))
    assert f_data.count == 1
