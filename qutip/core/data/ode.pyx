#cython: language_level=3
#cython: boundscheck=False, wraparound=False, initializedcheck=False

import numpy as np
cimport numpy as cnp
import inspect as _inspect

# QuTiP core imports
from qutip.core.data.base cimport idxint, Data
from qutip.core.data cimport dia, Dense, CSR, Dia
from .dispatch import Dispatcher as _Dispatcher


cnp.import_array()

__all__ = ["wrmn_error", "wrmn_error_dense", "wrmn_error_csr", "wrmn_error_dia"]


cdef int _check_shape(Data left, Data right) except -1 nogil:
    if left.shape[0] != right.shape[0] or left.shape[1] != right.shape[1]:
        raise ValueError(
            f"incompatible matrix shapes {left.shape} and {right.shape}"
        )
    return 0


cpdef double wrmn_error_dense(Dense diff, Dense state, double atol, double rtol) except -1:
    """
    Compute the weighted root mean square error norm:

        error = sqrt( 1/N * sum( ( |diff[i]| / (atol + rtol * |state[i]|))**2 ))
        
    This error norm is commonly used to estimate error in ODEs.
    """
    cdef double sum = 0.
    cdef size_t i, N = diff.shape[0] * diff.shape[1]
    cdef int dim1, dim2

    _check_shape(diff, state)

    if diff.fortran == state.fortran:
        for i in range(N):
            sum += (
                abs(diff.data[i])
                / (atol + rtol * abs(state.data[i]))
            )**2
    else:
        arr_diff = diff.as_ndarray()
        arr_state = state.as_ndarray()
        sum = ((np.abs(arr_diff) / (atol + rtol * np.abs(arr_state)))**2).sum()

    return (sum / N)**0.5


cpdef double wrmn_error_csr(CSR diff, CSR state, double atol, double rtol) except -1:
    """
    Compute the weighted root mean square norm for the ode error estimation

    error = sqrt( 1/N * sum( ( |diff[i]| / (atol + rtol * |state[i]|))**2 ))
    """
    _check_shape(diff, state)
    cdef double sum_ = 0.
    cdef size_t i, N = diff.shape[0] * diff.shape[1]

    cdef idxint row, ptr_diff, ptr_state, ptr_diff_max, ptr_state_max, nnz=0, col_diff, col_state
    cdef idxint ncols = diff.shape[1]
    cdef double complex val

    diff = diff.sort_indices()
    state = state.sort_indices()

    ptr_diff_max = ptr_state_max = 0
    for row in range(diff.shape[0]):
        ptr_diff = ptr_diff_max
        ptr_diff_max = diff.row_index[row + 1]
        ptr_state = ptr_state_max
        ptr_state_max = state.row_index[row + 1]

        # Use ncols as a sentinel value larger than any possible column index.
        col_diff = diff.col_index[ptr_diff] if ptr_diff < ptr_diff_max else ncols
        col_state = state.col_index[ptr_state] if ptr_state < ptr_state_max else ncols

        while ptr_diff < ptr_diff_max or ptr_state < ptr_state_max:
            if col_diff < col_state:
                # Element only in 'diff'
                sum_ += (abs(diff.data[ptr_diff]) / atol)**2
                ptr_diff += 1
                col_diff = diff.col_index[ptr_diff] if ptr_diff < ptr_diff_max else ncols
            elif col_state < col_diff:
                # Element only in 'state'.
                # The corresponding 'diff' element is zero, so the error
                # contribution is zero.
                ptr_state += 1
                col_state = state.col_index[ptr_state] if ptr_state < ptr_state_max else ncols
            else: # col_diff == col_state
                if col_diff == ncols: break # Both pointers are at the end of the row
                # Element in both 'diff' and 'state'
                sum_ += (
                    abs(diff.data[ptr_diff])
                    / (atol + rtol * abs(state.data[ptr_state]))
                )**2
                ptr_diff += 1
                ptr_state += 1
                col_diff = diff.col_index[ptr_diff] if ptr_diff < ptr_diff_max else ncols
                col_state = state.col_index[ptr_state] if ptr_state < ptr_state_max else ncols

    return (sum_ / N)**0.5


cpdef double wrmn_error_dia(Dia diff, Dia state, double atol, double rtol) except -1:
    """
    Compute the weighted root mean square norm for the ode error estimation

    error = sqrt( 1/N * sum( ( |diff[i]| / (atol + rtol * |state[i]|))**2 ))
    """
    _check_shape(diff, state)
    cdef double sum_ = 0.
    cdef idxint diag_diff=0, diag_state=0, size=diff.shape[1], i,
    cdef idxint offset, start, end
    cdef size_t idx_diff=0, idx_state=0, N = diff.shape[0] * diff.shape[1]

    if not diff._is_sorted():
        diff = dia.clean_dia(diff, False)
    if not state._is_sorted():
        state = dia.clean_dia(state, False)

    while diag_diff < diff.num_diag and diag_state < state.num_diag:
        if diff.offsets[diag_diff] == state.offsets[diag_state]:
            offset = diff.offsets[diag_diff]
            start = max(0, offset)
            end = min(diff.shape[1], diff.shape[0] + offset)
            for i in range(start, end):
                sum_ += (
                    abs(diff.data[idx_diff+i])
                    / (atol + rtol * abs(state.data[idx_state+i]))
                )**2
            idx_diff += size
            diag_diff += 1
            idx_state += size
            diag_state += 1
        elif diff.offsets[diag_diff] < state.offsets[diag_state]:
            offset = diff.offsets[diag_diff]
            start = max(0, offset)
            end = min(diff.shape[1], diff.shape[0] + offset)
            for i in range(start, end):
                sum_ += (abs(diff.data[idx_diff+i]) / atol )**2
            idx_diff += size
            diag_diff += 1
        else:
            # diag only in `state`. `diff` is 0 so no error contribution.
            idx_state += size
            diag_state += 1

    while diag_diff < diff.num_diag:
        # Leftover `diff` values.
        offset = diff.offsets[diag_diff]
        start = max(0, offset)
        end = min(diff.shape[1], diff.shape[0] + offset)
        for i in range(start, end):
            sum_ += (abs(diff.data[idx_diff+i]) / atol )**2
        idx_diff += size
        diag_diff += 1

    return (sum_ / N)**0.5


wrmn_error = _Dispatcher(
    _inspect.Signature([
        _inspect.Parameter('diff', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('state', _inspect.Parameter.POSITIONAL_ONLY),
        _inspect.Parameter('atol', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
        _inspect.Parameter('rtol', _inspect.Parameter.POSITIONAL_OR_KEYWORD),
    ]),
    name='wrmn_error',
    module=__name__,
    inputs=('diff', 'state'),
    out=False,
)
wrmn_error.__doc__ =\
    """
    Compute the weighted root mean square error norm:

        error = sqrt( 1/N * sum( ( |diff[i]| / (atol + rtol * |state[i]|))**2 ))
        
    This error norm is commonly used to estimate error in ODEs.

    Parameters
    ----------
    diff : Data
        Difference vector between approximations of the state after integration
        step.
    state : Data
        State at the end of the step.
    atol: float
        Absolsute tolerance
    rtol: float
        Relative tolerance

    Returns
    -------
    error: float
    """
wrmn_error.add_specialisations([
    (Dense, Dense, wrmn_error_dense),
    (CSR, CSR, wrmn_error_csr),
    (Dia, Dia, wrmn_error_dia),
], _defer=True)


del _inspect, _Dispatcher


cdef double cy_wrmn_error(Data diff, Data state, double atol, double rtol) except -1:
    """ c dispatcher to speed up ODE for dense states """
    if type(diff) is Dense and type(state) is Dense:
        return wrmn_error_dense(diff, state, atol, rtol)

    return wrmn_error(diff, state, atol, rtol)
