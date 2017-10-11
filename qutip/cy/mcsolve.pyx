# cython: profile=True
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
import numpy as np
import scipy.sparse as sp
from qutip.fastsparse import csr2fast
from qutip.qobj import Qobj
from qutip.cy.spmatfuncs import cy_expect_psi_csr, spmv_csr
from qutip.cy.spconvert import dense2D_to_fastcsr_cmode
from qutip.cy.dopri5 import ode_td_dopri
cimport numpy as np
cimport cython
from scipy.linalg.cython_blas cimport dznrm2 as raw_dznrm2
from qutip.cy.td_qobj_cy cimport cy_qobj


cdef int ONE = 1

cdef double dznrm2(complex[::1] psi):
    cdef int l = psi.shape[0]
    return raw_dznrm2(&l, <complex*>&psi[0], &ONE)

cdef complex[::1] normalize(complex[::1] psi):
    cdef int i, l = psi.shape[0]
    cdef double norm = dznrm2(psi)
    cdef complex[::1] out = np.empty(l, dtype=complex)
    for i in range(l):
        out[i] = psi[i] / norm
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
def cy_mc_run_ode(ODE, config, prng):
    cdef int i, ii, j, jj, k
    cdef double norm2_psi, norm2_prev, norm2_guess, t_prev, t_final, t_guess
    cdef np.ndarray[double, ndim=1] rand_vals
    cdef np.ndarray[complex, ndim=1] y_prev

    cdef np.ndarray[double, ndim=1] tlist = config.tlist
    cdef int num_times = len(tlist)

    cdef np.ndarray states_out

    if config.options.steady_state_average:
        states_out = np.zeros((1), dtype=object)
    else:
        states_out = np.zeros((num_times), dtype=object)
    temp = sp.csr_matrix(
        np.reshape(config.psi0, (config.psi0.shape[0], 1)),
        dtype=complex)
    temp = csr2fast(temp)
    if (config.options.average_states and
            not config.options.steady_state_average):
        # output is averaged states, so use dm
        states_out[0] = Qobj(temp*temp.H,
                             [config.psi0_dims[0],
                              config.psi0_dims[0]],
                             [config.psi0_shape[0],
                              config.psi0_shape[0]],
                             fast='mc-dm')
    elif (not config.options.average_states and
          not config.options.steady_state_average):
        # output is not averaged, so write state vectors
        states_out[0] = Qobj(temp, config.psi0_dims,
                             config.psi0_shape, fast='mc')
    elif config.options.steady_state_average:
        states_out[0] = temp * temp.H

    # PRE-GENERATE LIST FOR EXPECTATION VALUES
    expect_out = []
    for i in range(config.e_num):
        if config.e_ops_isherm[i]:
            # preallocate real array of zeros
            expect_out.append(np.zeros(num_times, dtype=float))
        else:
            # preallocate complex array of zeros
            expect_out.append(np.zeros(num_times, dtype=complex))

        expect_out[i][0] = \
            cy_expect_psi_csr(config.e_ops_data[i],
                              config.e_ops_ind[i],
                              config.e_ops_ptr[i],
                              config.psi0,
                              config.e_ops_isherm[i])

    collapse_times = []
    which_oper = []
    # first rand is collapse norm, second is which operator
    rand_vals = prng.rand(2)

    # make array for collapse operator inds
    cdef np.ndarray[long, ndim=1] cinds = np.arange(len(config.td_c_ops))

    c_ops_rhs = [c.get_rhs_func() for c in config.td_c_ops]
    c_expect_func = [c.get_expect_func() for c in config.td_n_ops]
    norm2_prev = dznrm2(ODE._y) ** 2
    # RUN ODE UNTIL EACH TIME IN TLIST
    for k in range(1, num_times):
        # ODE WHILE LOOP FOR INTEGRATE UP TO TIME TLIST[k]
        t_prev = ODE.t
        y_prev = ODE.y

        while t_prev < tlist[k]:
            # integrate up to tlist[k], one step at a time.
            ODE.integrate(tlist[k], step=1)
            if not ODE.successful():
                print(ODE.t, t_prev, tlist[k])
                print(ODE._integrator.call_args)
                raise Exception("ZVODE failed!")
            norm2_psi = dznrm2(ODE._y) ** 2
            if norm2_psi <= rand_vals[0]:
                # collapse has occured:
                # find collapse time to within specified tolerance
                # ------------------------------------------------
                ii = 0
                t_final = ODE.t
                while ii < config.norm_steps:
                    ii += 1
                    t_guess = t_prev + \
                        np.log(norm2_prev / rand_vals[0]) / \
                        np.log(norm2_prev / norm2_psi) * (t_final - t_prev)
                    ODE._y = y_prev
                    ODE.t = t_prev
                    ODE._integrator.call_args[3] = 1
                    ODE.integrate(t_guess, step=0)
                    if not ODE.successful():
                        raise Exception(
                            "ZVODE failed after adjusting step size!")
                    norm2_guess = dznrm2(ODE._y)**2
                    if (np.abs(rand_vals[0] - norm2_guess) <
                            config.norm_tol * rand_vals[0]):
                        norm2_psi = norm2_guess
                        t_prev = t_guess
                        y_prev = ODE.y
                        break
                    elif (norm2_guess < rand_vals[0]):
                        # t_guess is still > t_jump
                        t_final = t_guess
                        norm2_psi = norm2_guess
                    else:
                        # t_guess < t_jump
                        t_prev = t_guess
                        y_prev = ODE.y
                        norm2_prev = norm2_guess
                if ii > config.norm_steps:
                    raise Exception("Norm tolerance not reached. " +
                                    "Increase accuracy of ODE solver or " +
                                    "Options.norm_steps.")

                collapse_times.append(ODE.t)
                # some string based collapse operators
                n_dp = [expect(ODE.t, ODE._y, 1)
                        for expect in c_expect_func]
                # determine which operator does collapse and store it
                kk = np.cumsum(n_dp / np.sum(n_dp))
                j = cinds[kk >= rand_vals[1]][0]
                which_oper.append(j)
                state = c_ops_rhs[j](ODE.t, ODE._y)
                state = normalize(state)
                ODE.set_initial_value(state, t_prev)
                rand_vals = prng.rand(2)
            else:
                norm2_prev = norm2_psi
                t_prev = ODE.t
                y_prev = ODE.y

        # after while loop
        # ----------------
        out_psi = normalize(ODE._y)
        if config.e_num == 0 or config.options.store_states:
            out_psi_csr = dense2D_to_fastcsr_cmode(np.reshape(out_psi,
                                                   (out_psi.shape[0], 1)),
                                                    out_psi.shape[0], 1)
            if (config.options.average_states and
                    not config.options.steady_state_average):
                states_out[k] = Qobj(
                    out_psi_csr * out_psi_csr.H,
                    [config.psi0_dims[0], config.psi0_dims[0]],
                    [config.psi0_shape[0], config.psi0_shape[0]],
                    fast='mc-dm')

            elif config.options.steady_state_average:
                states_out[0] = (
                    states_out[0] +
                    (out_psi_csr * out_psi_csr.H))

            else:
                states_out[k] = Qobj(out_psi_csr, config.psi0_dims,
                                     config.psi0_shape, fast='mc')

        for jj in range(config.e_num):
            expect_out[jj][k] = cy_expect_psi_csr(
                config.e_ops_data[jj], config.e_ops_ind[jj],
                config.e_ops_ptr[jj], out_psi,
                config.e_ops_isherm[jj])

    return states_out, expect_out, collapse_times, which_oper


@cython.boundscheck(False)
@cython.wraparound(False)
def cy_mc_run_fast(config, prng):
    cdef int i, ii, j, jj, k
    cdef double t, tt
    cdef double[::1] rand_vals
    cdef double[::1] tlist = config.tlist
    cdef int num_times = len(tlist)

    cdef np.ndarray states_out

    cdef complex[::1] psi
    cdef double[::1] err = np.zeros(11)

    if config.options.steady_state_average:
        states_out = np.zeros((1), dtype=object)
    else:
        states_out = np.zeros((num_times), dtype=object)
    temp = sp.csr_matrix(
        np.reshape(config.psi0, (config.psi0.shape[0], 1)),
        dtype=complex)
    temp = csr2fast(temp)
    if (config.options.average_states and
            not config.options.steady_state_average):
        # output is averaged states, so use dm
        states_out[0] = Qobj(temp*temp.H,
                             [config.psi0_dims[0],
                              config.psi0_dims[0]],
                             [config.psi0_shape[0],
                              config.psi0_shape[0]],
                             fast='mc-dm')
    elif (not config.options.average_states and
          not config.options.steady_state_average):
        # output is not averaged, so write state vectors
        states_out[0] = Qobj(temp, config.psi0_dims,
                             config.psi0_shape, fast='mc')
    elif config.options.steady_state_average:
        states_out[0] = temp * temp.H
    # PRE-GENERATE LIST FOR EXPECTATION VALUES
    expect_out = []
    for i in range(config.e_num):
        if config.e_ops_isherm[i]:
            # preallocate real array of zeros
            expect_out.append(np.zeros(num_times, dtype=float))
        else:
            # preallocate complex array of zeros
            expect_out.append(np.zeros(num_times, dtype=complex))

        expect_out[i][0] = \
            cy_expect_psi_csr(config.e_ops_data[i],
                              config.e_ops_ind[i],
                              config.e_ops_ptr[i],
                              config.psi0,
                              config.e_ops_isherm[i])

    collapse_times = []
    which_oper = []
    # first rand is collapse norm, second is which operator
    rand_vals = prng.rand(2)

    # make array for collapse operator inds
    cdef np.ndarray[long, ndim=1] cinds = np.arange(len(config.td_c_ops))

    ODE = ode_td_dopri(len(config.psi0), config.H_td, config)

    c_ops_rhs = [c.get_rhs_func() for c in config.td_c_ops]
    c_expect_func = [c.get_expect_func() for c in config.td_n_ops]
    t = tlist[0]
    psi = np.array(config.psi0)
    # RUN ODE UNTIL EACH TIME IN TLIST
    for k in range(1, num_times):
        while t < tlist[k]:
            # integrate up to tlist[k], one step at a time.
            tt = ODE.integrate(t, tlist[k], rand_vals[0], psi, err)
            if(tt < 0):
                print("Error at time ", tlist[k])
                print(tt, np.array(err))
            t = tt
            if(t < tlist[k]):
                collapse_times.append(t)
                n_dp = [expect(t, psi, 1)
                        for expect in c_expect_func]

                kk = np.cumsum(n_dp / np.sum(n_dp))
                j = cinds[kk >= rand_vals[1]][0]
                which_oper.append(j)

                state = c_ops_rhs[j](t, psi)
                state = normalize(state)
                psi = state
                rand_vals = prng.rand(2)

        # after while loop
        # ----------------
        out_psi = normalize(psi)
        if config.e_num == 0 or config.options.store_states:
            out_psi_csr = dense2D_to_fastcsr_cmode(np.reshape(out_psi,
                                                   (out_psi.shape[0], 1)),
                                                   out_psi.shape[0], 1)
            if (config.options.average_states and
                    not config.options.steady_state_average):
                states_out[k] = Qobj(
                    out_psi_csr * out_psi_csr.H,
                    [config.psi0_dims[0], config.psi0_dims[0]],
                    [config.psi0_shape[0], config.psi0_shape[0]],
                    fast='mc-dm')

            elif config.options.steady_state_average:
                states_out[0] = (
                    states_out[0] +
                    (out_psi_csr * out_psi_csr.H))

            else:
                states_out[k] = Qobj(out_psi_csr, config.psi0_dims,
                                     config.psi0_shape, fast='mc')

        for jj in range(config.e_num):
            expect_out[jj][k] = cy_expect_psi_csr(
                config.e_ops_data[jj], config.e_ops_ind[jj],
                config.e_ops_ptr[jj], out_psi,
                config.e_ops_isherm[jj])

    return states_out, expect_out, collapse_times, which_oper
