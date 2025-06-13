import pytest

import numpy as np
from qutip.core import data as _data


@pytest.mark.parametrize('intype', _data.to.dtypes)
@pytest.mark.parametrize('outtype', _data.to.dtypes)
def test_concat(intype, outtype):
    block1 = np.full((2, 3), 1)
    data1 = _data.to(intype, _data.Dense(block1))

    block2 = np.full((2, 1), 2)
    data2 = _data.to(intype, _data.Dense(block2))

    data3 = None

    block4 = np.full((1, 1), 4)
    data4 = _data.to(intype, _data.Dense(block4))

    datas = np.array([[data1, data2], [data3, data4]], dtype=_data.Data)
    block_widths = np.array([3, 1], dtype=_data.base.idxint_dtype)
    block_heights = np.array([2, 1], dtype=_data.base.idxint_dtype)
    result = _data.concat[outtype](datas, block_widths, block_heights)

    np.testing.assert_array_equal(result.to_array(), np.array(
        [[1, 1, 1, 2],
         [1, 1, 1, 2],
         [0, 0, 0, 4]]
    ))
    assert result.__class__ == outtype


@pytest.mark.parametrize('intype', _data.to.dtypes)
@pytest.mark.parametrize('outtype', _data.to.dtypes)
def test_spconcat(intype, outtype):
    block1 = np.full((2, 3), 1)
    data1 = _data.to(intype, _data.Dense(block1))

    block2 = np.full((2, 1), 2)
    data2 = _data.to(intype, _data.Dense(block2))

    block3 = np.full((1, 1), 3)
    data3 = _data.to(intype, _data.Dense(block3))

    block4 = np.full((1, 1), 4)
    data4 = _data.to(intype, _data.Dense(block4))

    block_rows = np.array([0, 0, 1, 1], dtype=_data.base.idxint_dtype)
    block_cols = np.array([0, 3, 2, 3], dtype=_data.base.idxint_dtype)
    def generator(row, col):
        if row == 0 and col == 0:
            return data1
        elif row == 0 and col == 3:
            return data2
        elif row == 1 and col == 2:
            return data3
        elif row == 1 and col == 3:
            return data4
        else:
            assert False
    block_widths = np.array([3, 2, 1, 1], dtype=_data.base.idxint_dtype)
    block_heights = np.array([2, 1, 1, 2], dtype=_data.base.idxint_dtype)
    result = _data.spconcat[outtype](
        block_rows, block_cols, generator, block_widths, block_heights
    )

    np.testing.assert_array_equal(result.to_array(), np.array(
        [[1, 1, 1, 0, 0, 0, 2],
         [1, 1, 1, 0, 0, 0, 2],
         [0, 0, 0, 0, 0, 3, 4],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]]
    ))
    assert result.__class__ == outtype
