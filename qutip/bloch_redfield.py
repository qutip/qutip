# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, QuSTaR
#    All rights reserved.
#
#    Redistribution and use in source and binary forms, with or without
#    modification, are permitted provided that the following conditions are
#    met:
#
#    1. Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#
#    2. Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#    3. Neither the name of the QuTiP: Quantum Toolbox in Python nor the names
#       of its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
#    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
###############################################################################

__all__ = ['brmesolve', 'bloch_redfield_solve', 'bloch_redfield_tensor']

import numpy as np
import os
import types
import warnings
from functools import partial
import scipy.integrate
import scipy.sparse as sp
from qutip.qobj import Qobj, isket
from qutip.states import ket2dm
from qutip.superoperator import spre, spost, vec2mat, mat2vec, vec2mat_index
from qutip.expect import expect
from qutip.solver import Options, Result, config, _solver_safety_check
from qutip.cy.spmatfuncs import cy_ode_rhs
from qutip.cy.spconvert import dense2D_to_fastcsr_fmode
from qutip.superoperator import liouvillian
from qutip.interpolate import Cubic_Spline
from qutip.cy.spconvert import arr_coo2fast
from qutip.cy.br_codegen import BR_Codegen
from qutip.ui.progressbar import BaseProgressBar, TextProgressBar
from qutip.cy.utilities import _cython_build_cleanup
from qutip.expect import expect_rho_vec
from qutip.rhs_generate import _td_format_check
from qutip.cy.openmp.utilities import check_use_openmp
import qutip.settings as qset


# -----------------------------------------------------------------------------
# Solve the Bloch-Redfield master equation
#
def brmesolve(H, psi0, tlist, a_ops=[], e_ops=[], c_ops=[],
              args={}, use_secular=True, sec_cutoff = 0.1,
              tol=qset.atol,
              spectra_cb=None, options=None,
              progress_bar=None, _safe_mode=True):
    """
    Solves for the dynamics of a system using the Bloch-Redfield master equation,
    given an input Hamiltonian, Hermitian bath-coupling terms and their associated 
    spectrum functions, as well as possible Lindblad collapse operators.
              
    For time-independent systems, the Hamiltonian must be given as a Qobj,
    whereas the bath-coupling terms (a_ops), must be written as a nested list
    of operator - spectrum function pairs, where the frequency is specified by
    the `w` variable.
              
    *Example*

        a_ops = [[a+a.dag(),lambda w: 0.2*(w>=0)]] 
              
    For time-dependent systems, the Hamiltonian, a_ops, and Lindblad collapse
    operators (c_ops), can be specified in the QuTiP string-based time-dependent
    format.  For the a_op spectra, the frequency variable must be `w`, and the 
    string cannot contain any other variables other than the possibility of having
    a time-dependence through the time variable `t`:
                            
    *Example*

        a_ops = [[a+a.dag(), '0.2*exp(-t)*(w>=0)']]
              
    It is also possible to use Cubic_Spline objects for time-dependence.  In
    the case of a_ops, Cubic_Splines must be passed as a tuple:
              
    *Example*
              
        a_ops = [ [a+a.dag(), ( f(w), g(t)] ]
              
    where f(w) and g(t) are strings or Cubic_spline objects for the bath
    spectrum and time-dependence, respectively.
              
    Finally, if one has bath-couplimg terms of the form
    H = f(t)*a + conj[f(t)]*a.dag(), then the correct input format is
              
    *Example*
    
              a_ops = [ [(a,a.dag()), (f(w), g1(t), g2(t))],... ]

    where f(w) is the spectrum of the operators while g1(t) and g2(t)
    are the time-dependence of the operators `a` and `a.dag()`, respectively 
    
    Parameters
    ----------
    H : Qobj / list
        System Hamiltonian given as a Qobj or
        nested list in string-based format.

    psi0: Qobj
        Initial density matrix or state vector (ket).

    tlist : array_like
        List of times for evaluating evolution

    a_ops : list
        Nested list of Hermitian system operators that couple to 
        the bath degrees of freedom, along with their associated
        spectra.

    e_ops : list
        List of operators for which to evaluate expectation values.

    c_ops : list
        List of system collapse operators, or nested list in
        string-based format.

    args : dict (not implimented)
        Placeholder for future implementation, kept for API consistency.

    use_secular : bool {True}
        Use secular approximation when evaluating bath-coupling terms.
    
    sec_cutoff : float {0.1}
        Cutoff for secular approximation.
    
    tol : float {qutip.setttings.atol}
        Tolerance used for removing small values after 
        basis transformation.
              
    spectra_cb : list
        DEPRECIATED. Do not use.
    
    options : :class:`qutip.solver.Options`
        Options for the solver.
              
    progress_bar : BaseProgressBar
        Optional instance of BaseProgressBar, or a subclass thereof, for
        showing the progress of the simulation.

    Returns
    -------
    result: :class:`qutip.solver.Result`

        An instance of the class :class:`qutip.solver.Result`, which contains
        either an array of expectation values, for operators given in e_ops,
        or a list of states for the times specified by `tlist`.
    """
    if isinstance(c_ops, Qobj):
        c_ops = [c_ops]

    if isinstance(e_ops, Qobj):
        e_ops = [e_ops]

    if isinstance(e_ops, dict):
        e_ops_dict = e_ops
        e_ops = [e for e in e_ops.values()]
    else:
        e_ops_dict = None
    
    if not (spectra_cb is None):
        warnings.warn("The use of spectra_cb is depreciated.", DeprecationWarning)
        _a_ops = []
        for kk, a in enumerate(a_ops):
            _a_ops.append([a,spectra_cb[kk]])
        a_ops = _a_ops

    if _safe_mode:
        _solver_safety_check(H, psi0, a_ops+c_ops, e_ops, args)
    
    # check for type (if any) of time-dependent inputs
    _, n_func, n_str = _td_format_check(H, a_ops+c_ops)
    
    if progress_bar is None:
        progress_bar = BaseProgressBar()
    elif progress_bar is True:
        progress_bar = TextProgressBar()
        
    if options is None:
        options = Options()

    if (not options.rhs_reuse) or (not config.tdfunc):
        # reset config collapse and time-dependence flags to default values
        config.reset()
    
    #check if should use OPENMP
    check_use_openmp(options)
    
    if n_str == 0:
    
        R, ekets = bloch_redfield_tensor(H, a_ops, spectra_cb=None, c_ops=c_ops,
                    use_secular=use_secular, sec_cutoff=sec_cutoff)

        output = Result()
        output.solver = "brmesolve"
        output.times = tlist

        results = bloch_redfield_solve(R, ekets, psi0, tlist, e_ops, options,
                    progress_bar=progress_bar)

        if e_ops:
            output.expect = results
        else:
            output.states = results

        return output
        
    elif n_str != 0 and n_func == 0:
        output = _td_brmesolve(H, psi0, tlist, a_ops=a_ops, e_ops=e_ops, 
                        c_ops=c_ops, use_secular=use_secular, 
                        sec_cutoff=sec_cutoff,
                        tol=tol, options=options, 
                         progress_bar=progress_bar,
                         _safe_mode=_safe_mode)
                         
        return output
        
    else:
        raise Exception('Cannot mix func and str formats.')


# -----------------------------------------------------------------------------
# Evolution of the Bloch-Redfield master equation given the Bloch-Redfield
# tensor.
#
def bloch_redfield_solve(R, ekets, rho0, tlist, e_ops=[], options=None, progress_bar=None):
    """
    Evolve the ODEs defined by Bloch-Redfield master equation. The
    Bloch-Redfield tensor can be calculated by the function
    :func:`bloch_redfield_tensor`.

    Parameters
    ----------

    R : :class:`qutip.qobj`
        Bloch-Redfield tensor.

    ekets : array of :class:`qutip.qobj`
        Array of kets that make up a basis tranformation for the eigenbasis.

    rho0 : :class:`qutip.qobj`
        Initial density matrix.

    tlist : *list* / *array*
        List of times for :math:`t`.

    e_ops : list of :class:`qutip.qobj` / callback function
        List of operators for which to evaluate expectation values.

    options : :class:`qutip.Qdeoptions`
        Options for the ODE solver.

    Returns
    -------

    output: :class:`qutip.solver`

        An instance of the class :class:`qutip.solver`, which contains either
        an *array* of expectation values for the times specified by `tlist`.

    """

    if options is None:
        options = Options()

    if options.tidy:
        R.tidyup()

    if progress_bar is None:
        progress_bar = BaseProgressBar()
    elif progress_bar is True:
        progress_bar = TextProgressBar()
    
    #
    # check initial state
    #
    if isket(rho0):
        # Got a wave function as initial state: convert to density matrix.
        rho0 = rho0 * rho0.dag()

    #
    # prepare output array
    #
    n_tsteps = len(tlist)
    dt = tlist[1] - tlist[0]
    result_list = []

    #
    # transform the initial density matrix and the e_ops opterators to the
    # eigenbasis
    #
    rho_eb = rho0.transform(ekets)
    e_eb_ops = [e.transform(ekets) for e in e_ops]

    for e_eb in e_eb_ops:
        if e_eb.isherm:
            result_list.append(np.zeros(n_tsteps, dtype=float))
        else:
            result_list.append(np.zeros(n_tsteps, dtype=complex))

    #
    # setup integrator
    #
    initial_vector = mat2vec(rho_eb.full())
    r = scipy.integrate.ode(cy_ode_rhs)
    r.set_f_params(R.data.data, R.data.indices, R.data.indptr)
    r.set_integrator('zvode', method=options.method, order=options.order,
                     atol=options.atol, rtol=options.rtol,
                     nsteps=options.nsteps, first_step=options.first_step,
                     min_step=options.min_step, max_step=options.max_step)
    r.set_initial_value(initial_vector, tlist[0])

    #
    # start evolution
    #
    dt = np.diff(tlist)
    progress_bar.start(n_tsteps)
    for t_idx, _ in enumerate(tlist):
        progress_bar.update(t_idx)
        if not r.successful():
            break

        rho_eb.data = dense2D_to_fastcsr_fmode(vec2mat(r.y), rho0.shape[0], rho0.shape[1])

        # calculate all the expectation values, or output rho_eb if no
        # expectation value operators are given
        if e_ops:
            rho_eb_tmp = Qobj(rho_eb)
            for m, e in enumerate(e_eb_ops):
                result_list[m][t_idx] = expect(e, rho_eb_tmp)
        else:
            result_list.append(rho_eb.transform(ekets, True))

        if t_idx < n_tsteps - 1:
            r.integrate(r.t + dt[t_idx])
    progress_bar.finished()
    return result_list


# -----------------------------------------------------------------------------
# Functions for calculating the Bloch-Redfield tensor for a time-independent
# system.
#
def bloch_redfield_tensor(H, a_ops, spectra_cb=None, c_ops=[], use_secular=True, sec_cutoff=0.1):
    """
    Calculate the Bloch-Redfield tensor for a system given a set of operators
    and corresponding spectral functions that describes the system's coupling
    to its environment.

    .. note::

        This tensor generation requires a time-independent Hamiltonian.

    Parameters
    ----------

    H : :class:`qutip.qobj`
        System Hamiltonian.

    a_ops : list of :class:`qutip.qobj`
        List of system operators that couple to the environment.

    spectra_cb : list of callback functions
        List of callback functions that evaluate the noise power spectrum
        at a given frequency.

    c_ops : list of :class:`qutip.qobj`
        List of system collapse operators.

    use_secular : bool
        Flag (True of False) that indicates if the secular approximation should
        be used.
    
    sec_cutoff : float {0.1}
        Threshold for secular approximation.

    Returns
    -------

    R, kets: :class:`qutip.Qobj`, list of :class:`qutip.Qobj`

        R is the Bloch-Redfield tensor and kets is a list eigenstates of the
        Hamiltonian.

    """
    
    if not (spectra_cb is None):
        warnings.warn("The use of spectra_cb is depreciated.", DeprecationWarning)
        _a_ops = []
        for kk, a in enumerate(a_ops):
            _a_ops.append([a,spectra_cb[kk]])
        a_ops = _a_ops
    
    # Sanity checks for input parameters
    if not isinstance(H, Qobj):
        raise TypeError("H must be an instance of Qobj")

    for a in a_ops:
        if not isinstance(a[0], Qobj) or not a[0].isherm:
            raise TypeError("Operators in a_ops must be Hermitian Qobj.")

    if c_ops is None:
        c_ops = []

    # use the eigenbasis
    evals, ekets = H.eigenstates()

    N = len(evals)
    K = len(a_ops)
    
    #only Lindblad collapse terms
    if K==0:
        Heb = H.transform(ekets)
        L = liouvillian(Heb, c_ops=[c_op.transform(ekets) for c_op in c_ops])
        return L, ekets
    
    
    A = np.array([a_ops[k][0].transform(ekets).full() for k in range(K)])
    Jw = np.zeros((K, N, N), dtype=complex)

    # pre-calculate matrix elements and spectral densities
    # W[m,n] = real(evals[m] - evals[n])
    W = np.real(evals[:,np.newaxis] - evals[np.newaxis,:])

    for k in range(K):
        # do explicit loops here in case spectra_cb[k] can not deal with array arguments
        for n in range(N):
            for m in range(N):
                Jw[k, n, m] = a_ops[k][1](W[n, m])

    dw_min = np.abs(W[W.nonzero()]).min()

    # pre-calculate mapping between global index I and system indices a,b
    Iabs = np.empty((N*N,3),dtype=int)
    for I, Iab in enumerate(Iabs):
        # important: use [:] to change array values, instead of creating new variable Iab
        Iab[0]  = I
        Iab[1:] = vec2mat_index(N, I)

    # unitary part + dissipation from c_ops (if given):
    Heb = H.transform(ekets)
    L = liouvillian(Heb, c_ops=[c_op.transform(ekets) for c_op in c_ops])
    
    # dissipative part:
    rows = []
    cols = []
    data = []
    for I, a, b in Iabs:
        # only check use_secular once per I
        if use_secular:
            # only loop over those indices J which actually contribute
            Jcds = Iabs[np.where(np.abs(W[a, b] - W[Iabs[:,1], Iabs[:,2]]) < dw_min * sec_cutoff)]
        else:
            Jcds = Iabs
        for J, c, d in Jcds:
            elem = 0+0j
            # summed over k, i.e., each operator coupling the system to the environment
            elem += 0.5 * np.sum(A[:, a, c] * A[:, d, b] * (Jw[:, c, a] + Jw[:, d, b]))
            if b==d:
                #                  sum_{k,n} A[k, a, n] * A[k, n, c] * Jw[k, c, n])
                elem -= 0.5 * np.sum(A[:, a, :] * A[:, :, c] * Jw[:, c, :])
            if a==c:
                #                  sum_{k,n} A[k, d, n] * A[k, n, b] * Jw[k, d, n])
                elem -= 0.5 * np.sum(A[:, d, :] * A[:, :, b] * Jw[:, d, :])
            if elem != 0:
                rows.append(I)
                cols.append(J)
                data.append(elem)

    R = arr_coo2fast(np.array(data, dtype=complex),
                    np.array(rows, dtype=np.int32),
                    np.array(cols, dtype=np.int32), N**2, N**2)
    
    L.data = L.data + R
    
    return L, ekets




def _td_brmesolve(H, psi0, tlist, a_ops=[], e_ops=[], c_ops=[],
                 use_secular=True, sec_cutoff=0.1,
                 tol=qset.atol, options=None, 
                 progress_bar=None,_safe_mode=True):
    
    if isket(psi0):
        rho0 = ket2dm(psi0)
    else:
        rho0 = psi0
    nrows = rho0.shape[0]
    
    H_terms = []
    H_td_terms = []
    H_obj = []
    A_terms = []
    A_td_terms = []
    C_terms = []
    C_td_terms = []
    CA_obj = []
    spline_count = [0,0]
    coupled_ops = []
    coupled_lengths = []
    coupled_spectra = []
    
    if isinstance(H, Qobj):
        H_terms.append(H.full('f'))
        H_td_terms.append('1')
    else: 
        for kk, h in enumerate(H):
            if isinstance(h, Qobj):
                H_terms.append(h.full('f'))
                H_td_terms.append('1')
            elif isinstance(h, list):
                H_terms.append(h[0].full('f'))
                if isinstance(h[1], Cubic_Spline):
                    H_obj.append(h[1].coeffs)
                    spline_count[0] += 1
                H_td_terms.append(h[1])
            else:
                raise Exception('Invalid Hamiltonian specification.')
    
            
    for kk, c in enumerate(c_ops):
        if isinstance(c, Qobj):
            C_terms.append(c.full('f'))
            C_td_terms.append('1')
        elif isinstance(c, list):
            C_terms.append(c[0].full('f'))
            if isinstance(c[1], Cubic_Spline):
                CA_obj.append(c[1].coeffs)
                spline_count[0] += 1
            C_td_terms.append(c[1])
        else:
            raise Exception('Invalid collape operator specification.')
            
    coupled_offset = 0
    for kk, a in enumerate(a_ops):
        if isinstance(a, list):
            if isinstance(a[0], Qobj):
                A_terms.append(a[0].full('f'))
                A_td_terms.append(a[1])
                if isinstance(a[1], tuple):
                    if not len(a[1])==2:
                       raise Exception('Tuple must be len=2.')
                    if isinstance(a[1][0],Cubic_Spline):
                        spline_count[1] += 1
                    if isinstance(a[1][1],Cubic_Spline):
                        spline_count[1] += 1
            elif isinstance(a[0], tuple):
                if not isinstance(a[1], tuple):
                    raise Exception('Invalid bath-coupling specification.')
                if (len(a[0])+1) != len(a[1]):
                    raise Exception('BR a_ops tuple lengths not compatible.')
                
                coupled_ops.append(kk+coupled_offset)
                coupled_lengths.append(len(a[0]))
                coupled_spectra.append(a[1][0])
                coupled_offset += len(a[0])-1
                if isinstance(a[1][0],Cubic_Spline):
                    spline_count[1] += 1
                
                for nn, _a in enumerate(a[0]):
                    A_terms.append(_a.full('f'))
                    A_td_terms.append(a[1][nn+1])
                    if isinstance(a[1][nn+1],Cubic_Spline):
                        CA_obj.append(a[1][nn+1].coeffs)
                        spline_count[1] += 1
                                
        else:
            raise Exception('Invalid bath-coupling specification.')
            
    
    string_list = []
    for kk,_ in enumerate(H_td_terms):
        string_list.append("H_terms[{0}]".format(kk))
    for kk,_ in enumerate(H_obj):
        string_list.append("H_obj[{0}]".format(kk))
    for kk,_ in enumerate(C_td_terms):
        string_list.append("C_terms[{0}]".format(kk))
    for kk,_ in enumerate(CA_obj):
        string_list.append("CA_obj[{0}]".format(kk))
    for kk,_ in enumerate(A_td_terms):
        string_list.append("A_terms[{0}]".format(kk))
    #Add nrows to parameters
    string_list.append('nrows')
    parameter_string = ",".join(string_list)
    
    #
    # generate and compile new cython code if necessary
    #
    if not options.rhs_reuse or config.tdfunc is None:
        if options.rhs_filename is None:
            config.tdname = "rhs" + str(os.getpid()) + str(config.cgen_num)
        else:
            config.tdname = opt.rhs_filename
        cgen = BR_Codegen(h_terms=len(H_terms), 
                    h_td_terms=H_td_terms, h_obj=H_obj,
                    c_terms=len(C_terms), 
                    c_td_terms=C_td_terms, c_obj=CA_obj,
                    a_terms=len(A_terms), a_td_terms=A_td_terms,
                    spline_count=spline_count,
                    coupled_ops = coupled_ops,
                    coupled_lengths = coupled_lengths,
                    coupled_spectra = coupled_spectra,
                    config=config, sparse=False,
                    use_secular = use_secular,
                    sec_cutoff = sec_cutoff,
                    use_openmp=options.use_openmp, 
                    omp_thresh=qset.openmp_thresh if qset.has_openmp else None,
                    omp_threads=options.num_cpus, 
                    atol=tol)
        
        cgen.generate(config.tdname + ".pyx")
        code = compile('from ' + config.tdname + ' import cy_td_ode_rhs',
                       '<string>', 'exec')
        exec(code, globals())
        config.tdfunc = cy_td_ode_rhs
    
    initial_vector = mat2vec(rho0.full()).ravel()
    
    _ode = scipy.integrate.ode(config.tdfunc)
    code = compile('_ode.set_f_params(' + parameter_string + ')',
                    '<string>', 'exec')
    _ode.set_integrator('zvode', method=options.method, 
                    order=options.order, atol=options.atol, 
                    rtol=options.rtol, nsteps=options.nsteps,
                    first_step=options.first_step, 
                    min_step=options.min_step,
                    max_step=options.max_step)
    _ode.set_initial_value(initial_vector, tlist[0])
    exec(code, locals())
    
    #
    # prepare output array
    #
    n_tsteps = len(tlist)
    e_sops_data = []

    output = Result()
    output.solver = "brmesolve"
    output.times = tlist

    if options.store_states:
        output.states = []

    if isinstance(e_ops, types.FunctionType):
        n_expt_op = 0
        expt_callback = True

    elif isinstance(e_ops, list):
        n_expt_op = len(e_ops)
        expt_callback = False

        if n_expt_op == 0:
            # fall back on storing states
            output.states = []
            options.store_states = True
        else:
            output.expect = []
            output.num_expect = n_expt_op
            for op in e_ops:
                e_sops_data.append(spre(op).data)
                if op.isherm:
                    output.expect.append(np.zeros(n_tsteps))
                else:
                    output.expect.append(np.zeros(n_tsteps, dtype=complex))

    else:
        raise TypeError("Expectation parameter must be a list or a function")

    #
    # start evolution
    #
    progress_bar.start(n_tsteps)

    rho = Qobj(rho0)

    dt = np.diff(tlist)
    for t_idx, t in enumerate(tlist):
        progress_bar.update(t_idx)

        if not _ode.successful():
            raise Exception("ODE integration error: Try to increase "
                            "the allowed number of substeps by increasing "
                            "the nsteps parameter in the Options class.")

        if options.store_states or expt_callback:
            rho.data = dense2D_to_fastcsr_fmode(vec2mat(_ode.y), rho.shape[0], rho.shape[1])

            if options.store_states:
                output.states.append(Qobj(rho, isherm=True))

            if expt_callback:
                # use callback method
                e_ops(t, rho)

        for m in range(n_expt_op):
            if output.expect[m].dtype == complex:
                output.expect[m][t_idx] = expect_rho_vec(e_sops_data[m],
                                                         _ode.y, 0)
            else:
                output.expect[m][t_idx] = expect_rho_vec(e_sops_data[m],
                                                         _ode.y, 1)

        if t_idx < n_tsteps - 1:
            _ode.integrate(_ode.t + dt[t_idx])

    progress_bar.finished()

    if (not options.rhs_reuse) and (config.tdname is not None):
        _cython_build_cleanup(config.tdname)
    
    if options.store_final_state:
        rho.data = dense2D_to_fastcsr_fmode(vec2mat(_ode.y), rho.shape[0], rho.shape[1])
        output.final_state = Qobj(rho, dims=rho0.dims, isherm=True)

    return output