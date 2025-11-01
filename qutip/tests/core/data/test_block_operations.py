import pytest
import numpy as np
from qutip.core import data as _data
from qutip.core.data import csr, Dense


@pytest.mark.parametrize('outtype', _data.to.dtypes)
def test_empty_concat_blocks(outtype):
    """concat_blocks with no blocks should return a zero matrix"""
    block_rows = np.array([], dtype=_data.base.idxint_dtype)
    block_cols = np.array([], dtype=_data.base.idxint_dtype)
    blocks = np.array([], dtype=_data.Data)
    block_widths = np.array([2, 3], dtype=_data.base.idxint_dtype)
    block_heights = np.array([4], dtype=_data.base.idxint_dtype)

    result = _data.concat_blocks(
        block_rows, block_cols, blocks, block_widths, block_heights,
        dtype=outtype
    )

    np.testing.assert_array_equal(result.to_array(), np.zeros((4, 5)))
    assert result.__class__ == outtype


@pytest.mark.parametrize('intype', _data.to.dtypes)
@pytest.mark.parametrize('outtype', _data.to.dtypes)
def test_concat_blocks(intype, outtype):
    """more complex example of concat_blocks"""
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
    blocks = np.array([data1, data2, data3, data4], dtype=_data.Data)
    block_widths = np.array([3, 2, 1, 1], dtype=_data.base.idxint_dtype)
    block_heights = np.array([2, 1, 1, 2], dtype=_data.base.idxint_dtype)
    result = _data.concat_blocks(
        block_rows, block_cols, blocks, block_widths, block_heights,
        dtype=outtype
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



@pytest.mark.parametrize('outtype', _data.to.dtypes)
def test_concat_blocks_validation(outtype):
    """validation checks in concat_blocks"""
    block = Dense(np.array([[1]], dtype=complex))

    # Test mismatched lengths of block_rows and block_cols
    with pytest.raises(ValueError, match="must have the same length"):
        _data.concat_blocks(
            np.array([0], dtype=_data.base.idxint_dtype),
            np.array([0, 1], dtype=_data.base.idxint_dtype),
            np.array([block], dtype=_data.Data),
            np.array([2, 3], dtype=_data.base.idxint_dtype),
            np.array([2, 3], dtype=_data.base.idxint_dtype),
            dtype=outtype
        )

    # Test mismatched lengths of blocks
    with pytest.raises(ValueError, match="must have the same length"):
        _data.concat_blocks(
            np.array([0, 0], dtype=_data.base.idxint_dtype),
            np.array([0, 1], dtype=_data.base.idxint_dtype),
            np.array([block], dtype=_data.Data),
            np.array([2, 3], dtype=_data.base.idxint_dtype),
            np.array([2, 3], dtype=_data.base.idxint_dtype),
            dtype=outtype
        )

    # Test incompatible block size
    with pytest.raises(ValueError, match="does not have the correct shape"):
        _data.concat_blocks(
            np.array([1], dtype=_data.base.idxint_dtype),
            np.array([1], dtype=_data.base.idxint_dtype),
            np.array([block], dtype=_data.Data),
            np.array([2, 3], dtype=_data.base.idxint_dtype),
            np.array([2, 3], dtype=_data.base.idxint_dtype),
            dtype=outtype
        )


def test_concat_blocks_csr():
    """tests specific to the csr implementation"""
    block = _data.one_element_csr((1, 1), (0, 0))

    # Test correct error message if arrays are not sorted
    with pytest.raises(ValueError, match="must be sorted"):
        _data.concat_blocks_csr(
            np.array([1, 0], dtype=_data.base.idxint_dtype),
            np.array([0, 1], dtype=_data.base.idxint_dtype),
            np.array([block, block], dtype=_data.Data),
            np.array([1, 1], dtype=_data.base.idxint_dtype),
            np.array([1, 1], dtype=_data.base.idxint_dtype),
        )
    with pytest.raises(ValueError, match="must be sorted"):
        _data.concat_blocks_csr(
            np.array([1, 1], dtype=_data.base.idxint_dtype),
            np.array([1, 0], dtype=_data.base.idxint_dtype),
            np.array([block, block], dtype=_data.Data),
            np.array([1, 1], dtype=_data.base.idxint_dtype),
            np.array([1, 1], dtype=_data.base.idxint_dtype),
        )
    with pytest.raises(ValueError, match="must be sorted"):
        _data.concat_blocks_csr(
            np.array([1, 0], dtype=_data.base.idxint_dtype),
            np.array([1, 0], dtype=_data.base.idxint_dtype),
            np.array([block, block], dtype=_data.Data),
            np.array([1, 1], dtype=_data.base.idxint_dtype),
            np.array([1, 1], dtype=_data.base.idxint_dtype),
        )

    # Test no segfault if input is csr.empty
    # (users are not expected to be exposed to csr.empty directly, but it is
    # good to avoid segfaults, so we test this here explicitly)
    result = _data.concat_blocks_csr(
        np.array([0, 0, 1, 1], dtype=_data.base.idxint_dtype),
        np.array([0, 1, 0, 1], dtype=_data.base.idxint_dtype),
        np.array([csr.identity(2), csr.empty(2, 2, 0), csr.empty(2, 2, 0), csr.identity(2)], dtype=_data.Data),
        np.array([2, 2], dtype=_data.base.idxint_dtype),
        np.array([2, 2], dtype=_data.base.idxint_dtype),
    )
    assert result == csr.identity(4)
    assert csr.nnz(result) == 4


@pytest.mark.parametrize('intype', _data.to.dtypes)
@pytest.mark.parametrize('outtype', _data.to.dtypes)
def test_slice(intype, outtype):
    original = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ], dtype=complex)
    data = _data.to(intype, _data.Dense(original))

    result = _data.slice(data, 1, 3, 0, 2, dtype=outtype)
    expected = np.array([
        [5, 6],
        [9, 10]
    ], dtype=complex)

    np.testing.assert_array_equal(result.to_array(), expected)
    assert result.__class__ == outtype


@pytest.mark.parametrize('intype', _data.to.dtypes)
def test_slice_validation(intype):
    """validation of slice parameters"""
    data = _data.zeros(3, 3, dtype=intype)

    # Test invalid slice indices
    with pytest.raises(IndexError):
        _data.slice(data, -1, 2, 0, 2)
    with pytest.raises(IndexError):
        _data.slice(data, 0, 4, 0, 2)
    with pytest.raises(IndexError):
        _data.slice(data, 1, 1, 0, 2)


@pytest.mark.parametrize('intype', _data.to.dtypes)
@pytest.mark.parametrize('outtype', _data.to.dtypes)
@pytest.mark.parametrize('data_array', [
    pytest.param(np.zeros((4, 4), dtype=complex), id='zeros'),
    pytest.param(np.ones((4, 4), dtype=complex), id='full'),
    pytest.param(np.array([[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]], dtype=complex), id='pattern1'),
    pytest.param(np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]], dtype=complex), id='pattern2'),
])
@pytest.mark.parametrize('block_array', [
    pytest.param(np.array([[1, 2], [3, 4]], dtype=complex), id='block1'),
    pytest.param(np.zeros((1, 1)), id='zero')
])
def test_insert_block(intype, outtype, data_array, block_array):
    data = _data.to(intype, _data.Dense(data_array))
    block = _data.to(intype, _data.Dense(block_array))

    result = _data.insert_block(data, block, 1, 1, dtype=outtype)
    expected = np.copy(data_array)
    expected[1:1 + block_array.shape[0], 1:1 + block_array.shape[1]] = block_array

    assert result.__class__ == outtype
    np.testing.assert_array_equal(result.to_array(), expected)


@pytest.mark.parametrize('intype', _data.to.dtypes)
def test_insert_block_validation(intype):
    """validation of insert_block parameters"""
    data = _data.zeros(2, 2, dtype=intype)
    block = _data.to(intype, _data.Dense(np.array([[1, 2], [3, 4]], dtype=complex)))

    # Test block too large
    with pytest.raises(IndexError, match="doesn't fit"):
        _data.insert_block(data, block, 1, 0)

    # Test negative indices
    with pytest.raises(IndexError, match="doesn't fit"):
        _data.insert_block(data, block, -1, 0)
    with pytest.raises(IndexError, match="doesn't fit"):
        _data.insert_block(data, block, 0, -1)