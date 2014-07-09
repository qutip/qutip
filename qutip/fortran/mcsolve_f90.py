# This file is part of QuTiP: Quantum Toolbox in Python.
#
#    Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
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
#
#    Original Code by Arne Grimsmo (2012): github.com/arnelg/qutipf90mc
###############################################################################

import numpy as np

from qutip.fortran import qutraj_run as qtf90
from qutip.qobj import Qobj
from qutip.mcsolve import _mc_data_config
from qutip.solver import Options, Result, config
from qutip.settings import debug
import qutip.settings

if debug:
    import inspect
    import os

# Working precision
wpr = np.dtype(np.float64)
wpc = np.dtype(np.complex128)


def mcsolve_f90(H, psi0, tlist, c_ops, e_ops, ntraj=None,
                options=Options(), sparse_dms=True, serial=False,
                ptrace_sel=[], calc_entropy=False):
    """
    Monte-Carlo wave function solver with fortran 90 backend.
    Usage is identical to qutip.mcsolve, for problems without explicit
    time-dependence, and with some optional input:

    Parameters
    ----------
    H : qobj
        System Hamiltonian.
    psi0 : qobj
        Initial state vector
    tlist : array_like
        Times at which results are recorded.
    ntraj : int
        Number of trajectories to run.
    c_ops : array_like
        ``list`` or ``array`` of collapse operators.
    e_ops : array_like
        ``list`` or ``array`` of operators for calculating expectation values.
    options : Options
        Instance of solver options.
    sparse_dms : boolean
        If averaged density matrices are returned, they will be stored as
        sparse (Compressed Row Format) matrices during computation if
        sparse_dms = True (default), and dense matrices otherwise. Dense
        matrices might be preferable for smaller systems.
    serial : boolean
        If True (default is False) the solver will not make use of the
        multiprocessing module, and simply run in serial.
    ptrace_sel: list
        This optional argument specifies a list of components to keep when
        returning a partially traced density matrix. This can be convenient for
        large systems where memory becomes a problem, but you are only
        interested in parts of the density matrix.
    calc_entropy : boolean
        If ptrace_sel is specified, calc_entropy=True will have the solver
        return the averaged entropy over trajectories in results.entropy. This
        can be interpreted as a measure of entanglement. See Phys. Rev. Lett.
        93, 120408 (2004), Phys. Rev. A 86, 022310 (2012).

    Returns
    -------
    results : Result
        Object storing all results from simulation.

    """
    if ntraj is None:
        ntraj = options.ntraj

    if psi0.type != 'ket':
        raise Exception("Initial state must be a state vector.")
    config.options = options
    # set num_cpus to the value given in qutip.settings
    # if none in Options
    if not config.options.num_cpus:
        config.options.num_cpus = qutip.settings.num_cpus
    # set initial value data
    if options.tidy:
        config.psi0 = psi0.tidyup(options.atol).full()
    else:
        config.psi0 = psi0.full()
    config.psi0_dims = psi0.dims
    config.psi0_shape = psi0.shape
    # set general items
    config.tlist = tlist
    if isinstance(ntraj, (list, np.ndarray)):
        raise Exception("ntraj as list argument is not supported.")
    else:
        config.ntraj = ntraj
        # ntraj_list = [ntraj]
    # set norm finding constants
    config.norm_tol = options.norm_tol
    config.norm_steps = options.norm_steps

    if not options.rhs_reuse:
        config.soft_reset()
        # no time dependence
        config.tflag = 0
        # check for collapse operators
        if len(c_ops) > 0:
            config.cflag = 1
        else:
            config.cflag = 0
        # Configure data
        _mc_data_config(H, psi0, [], c_ops, [], [], e_ops, options, config)

    # Load Monte Carlo class
    mc = _MC_class()
    # Set solver type
    if (options.method == 'adams'):
        mc.mf = 10
    elif (options.method == 'bdf'):
        mc.mf = 22
    else:
        if debug:
            print('Unrecognized method for ode solver, using "adams".')
        mc.mf = 10
    # store ket and density matrix dims and shape for convenience
    mc.psi0_dims = psi0.dims
    mc.psi0_shape = psi0.shape
    mc.dm_dims = (psi0 * psi0.dag()).dims
    mc.dm_shape = (psi0 * psi0.dag()).shape
    # use sparse density matrices during computation?
    mc.sparse_dms = sparse_dms
    # run in serial?
    mc.serial_run = serial or (ntraj == 1)
    # are we doing a partial trace for returned states?
    mc.ptrace_sel = ptrace_sel
    if (ptrace_sel != []):
        if debug:
            print("ptrace_sel set to " + str(ptrace_sel))
            print("We are using dense density matrices during computation " +
                  "when performing partial trace. Setting sparse_dms = False")
            print("This feature is experimental.")
        mc.sparse_dms = False
        mc.dm_dims = psi0.ptrace(ptrace_sel).dims
        mc.dm_shape = psi0.ptrace(ptrace_sel).shape
    if (calc_entropy):
        if (ptrace_sel == []):
            if debug:
                print("calc_entropy = True, but ptrace_sel = []. Please set " +
                      "a list of components to keep when calculating average" +
                      " entropy of reduced density matrix in ptrace_sel. " +
                      "Setting calc_entropy = False.")
            calc_entropy = False
        mc.calc_entropy = calc_entropy

    # construct output Result object
    output = Result()

    # Run
    mc.run()
    output.states = mc.sol.states
    output.expect = mc.sol.expect
    output.col_times = mc.sol.col_times
    output.col_which = mc.sol.col_which
    if (hasattr(mc.sol, 'entropy')):
        output.entropy = mc.sol.entropy

    output.solver = 'Fortran 90 Monte Carlo solver'
    # simulation parameters
    output.times = config.tlist
    output.num_expect = config.e_num
    output.num_collapse = config.c_num
    output.ntraj = config.ntraj

    return output


class _MC_class():
    def __init__(self):
        self.cpus = config.options.num_cpus
        self.nprocs = self.cpus
        self.sol = Result()
        self.mf = 10
        # If returning density matrices, return as sparse or dense?
        self.sparse_dms = True
        # Run in serial?
        self.serial_run = False
        self.ntraj = config.ntraj
        self.ntrajs = []
        self.seed = None
        self.psi0_dims = None
        self.psi0_shape = None
        self.dm_dims = None
        self.dm_shape = None
        self.unravel_type = 2
        self.ptrace_sel = []
        self.calc_entropy = False

    def parallel(self):
        from multiprocessing import Process, Queue, JoinableQueue

        if debug:
            print(inspect.stack()[0][3])

        self.ntrajs = []
        for i in range(self.cpus):
            self.ntrajs.append(min(int(np.floor(float(self.ntraj)
                                                / self.cpus)),
                                   self.ntraj - sum(self.ntrajs)))
        cnt = sum(self.ntrajs)
        while cnt < self.ntraj:
            for i in range(self.cpus):
                self.ntrajs[i] += 1
                cnt += 1
                if (cnt >= self.ntraj):
                    break
        self.ntrajs = np.array(self.ntrajs)
        self.ntrajs = self.ntrajs[np.where(self.ntrajs > 0)]
        self.nprocs = len(self.ntrajs)
        sols = []
        processes = []
        resq = JoinableQueue()
        resq.join()

        if debug:
            print("Number of cpus: " + str(self.cpus))
            print("Trying to start " + str(self.nprocs) + " process(es).")
            print("Number of trajectories for each process: " +
                  str(self.ntrajs))

        for i in range(self.nprocs):
            p = Process(target=self.evolve_serial,
                        args=((resq, self.ntrajs[i], i, self.seed * (i + 1)),))
            p.start()
            processes.append(p)
        cnt = 0

        while True:
            try:
                sols.append(resq.get())
                resq.task_done()
                cnt += 1
                if (cnt >= self.nprocs):
                    break
            except KeyboardInterrupt:
                break
            except:
                pass

        resq.join()
        for proc in processes:
            try:
                proc.join()
            except KeyboardInterrupt:
                if debug:
                    print("Cancel thread on keyboard interrupt")
                proc.terminate()
                proc.join()
        resq.close()
        return sols

    def serial(self):

        if debug:
            print(inspect.stack()[0][3])

        self.nprocs = 1
        self.ntrajs = [self.ntraj]
        if debug:
            print("Running in serial.")
            print("Number of trajectories: " + str(self.ntraj))
        sol = self.evolve_serial((0, self.ntraj, 0, self.seed))
        return [sol]

    def run(self):

        if debug:
            print(inspect.stack()[0][3])

        from numpy.random import random_integers
        if (config.c_num == 0):
            # force one trajectory if no collapse operators
            config.ntraj = 1
            self.ntraj = 1
            # Set unravel_type to 1 to integrate without collapses
            self.unravel_type = 1
            if (config.e_num == 0):
                # If we are returning states, and there are no
                # collapse operators, set average_states to False to return
                # ket vectors instead of density matrices
                config.options.average_states = False
        # generate a random seed, useful if e.g. running with MPI
        self.seed = random_integers(1e8)
        if (self.serial_run):
            # run in serial
            sols = self.serial()
        else:
            # run in paralell
            sols = self.parallel()
        # gather data
        self.sol = _gather(sols)

    def evolve_serial(self, args):

        if debug:
            print(inspect.stack()[0][3] + ":" + str(os.getpid()))

        # run ntraj trajectories for one process via fortran
        # get args
        queue, ntraj, instanceno, rngseed = args
        # initialize the problem in fortran
        _init_tlist()
        _init_psi0()
        if (self.ptrace_sel != []):
            _init_ptrace_stuff(self.ptrace_sel)
        _init_hamilt()
        if (config.c_num != 0):
            _init_c_ops()
        if (config.e_num != 0):
            _init_e_ops()
        # set options
        qtf90.qutraj_run.n_c_ops = config.c_num
        qtf90.qutraj_run.n_e_ops = config.e_num
        qtf90.qutraj_run.ntraj = ntraj
        qtf90.qutraj_run.unravel_type = self.unravel_type
        qtf90.qutraj_run.average_states = config.options.average_states
        qtf90.qutraj_run.average_expect = config.options.average_expect
        qtf90.qutraj_run.init_result(config.psi0_shape[0],
                                     config.options.atol,
                                     config.options.rtol, mf=self.mf,
                                     norm_steps=config.norm_steps,
                                     norm_tol=config.norm_tol)
        # set optional arguments
        qtf90.qutraj_run.order = config.options.order
        qtf90.qutraj_run.nsteps = config.options.nsteps
        qtf90.qutraj_run.first_step = config.options.first_step
        qtf90.qutraj_run.min_step = config.options.min_step
        qtf90.qutraj_run.max_step = config.options.max_step
        qtf90.qutraj_run.norm_steps = config.options.norm_steps
        qtf90.qutraj_run.norm_tol = config.options.norm_tol
        # use sparse density matrices during computation?
        qtf90.qutraj_run.rho_return_sparse = self.sparse_dms
        # calculate entropy of reduced density matrice?
        qtf90.qutraj_run.calc_entropy = self.calc_entropy
        # run
        show_progress = 1 if debug else 0
        qtf90.qutraj_run.evolve(instanceno, rngseed, show_progress)

        # construct Result instance
        sol = Result()
        sol.ntraj = ntraj
        # sol.col_times = qtf90.qutraj_run.col_times
        # sol.col_which = qtf90.qutraj_run.col_which-1
        sol.col_times, sol.col_which = self.get_collapses(ntraj)
        if (config.e_num == 0):
            sol.states = self.get_states(len(config.tlist), ntraj)
        else:
            sol.expect = self.get_expect(len(config.tlist), ntraj)
        if (self.calc_entropy):
            sol.entropy = self.get_entropy(len(config.tlist))

        if (not self.serial_run):
            # put to queue
            queue.put(sol)
            queue.join()

        # deallocate stuff
        # finalize()
        return sol

    # Routines for retrieving data data from fortran
    def get_collapses(self, ntraj):

        if debug:
            print(inspect.stack()[0][3])

        col_times = np.zeros((ntraj), dtype=np.ndarray)
        col_which = np.zeros((ntraj), dtype=np.ndarray)
        if (config.c_num == 0):
            # no collapses
            return col_times, col_which
        for i in range(ntraj):
            qtf90.qutraj_run.get_collapses(i + 1)
            times = qtf90.qutraj_run.col_times
            which = qtf90.qutraj_run.col_which
            if times is None:
                times = np.array([])
            if which is None:
                which = np.array([])
            else:
                which = which - 1
            col_times[i] = np.array(times, copy=True)
            col_which[i] = np.array(which, copy=True)
        return col_times, col_which

    def get_states(self, nstep, ntraj):

        if debug:
            print(inspect.stack()[0][3])

        from scipy.sparse import csr_matrix
        if (config.options.average_states):
            states = np.array([Qobj()] * nstep)
            if (self.sparse_dms):
                # averaged sparse density matrices
                for i in range(nstep):
                    qtf90.qutraj_run.get_rho_sparse(i + 1)
                    val = qtf90.qutraj_run.csr_val
                    col = qtf90.qutraj_run.csr_col - 1
                    ptr = qtf90.qutraj_run.csr_ptr - 1
                    m = qtf90.qutraj_run.csr_nrows
                    k = qtf90.qutraj_run.csr_ncols
                    states[i] = Qobj(csr_matrix((val, col, ptr),
                                                (m, k)).toarray(),
                                     dims=self.dm_dims, shape=self.dm_shape)
            else:
                # averaged dense density matrices
                for i in range(nstep):
                    states[i] = Qobj(qtf90.qutraj_run.sol[0, i, :, :],
                                     dims=self.dm_dims, shape=self.dm_shape)
        else:
            # all trajectories as kets
            if (ntraj == 1):
                states = np.array([Qobj()] * nstep)
                for i in range(nstep):
                    states[i] = Qobj(np.matrix(
                        qtf90.qutraj_run.sol[0, 0, i, :]).transpose(),
                        dims=self.psi0_dims, shape=self.psi0_shape)
            else:
                states = np.array([np.array([Qobj()] * nstep)] * ntraj)
                for traj in range(ntraj):
                    for i in range(nstep):
                        states[traj][i] = Qobj(np.matrix(
                            qtf90.qutraj_run.sol[0, traj, i, :]).transpose(),
                            dims=self.psi0_dims, shape=self.psi0_shape)
        return states

    def get_expect(self, nstep, ntraj):

        if debug:
            print(inspect.stack()[0][3])

        if (config.options.average_expect):
            expect = []
            for j in range(config.e_num):
                if config.e_ops_isherm[j]:
                    expect += [np.real(qtf90.qutraj_run.sol[j, 0, :, 0])]
                else:
                    expect += [qtf90.qutraj_run.sol[j, 0, :, 0]]
        else:
            expect = np.array([[np.array([0. + 0.j] * nstep)] *
                               config.e_num] * ntraj)
            for j in range(config.e_num):
                expect[:, j, :] = qtf90.qutraj_run.sol[j, :, :, 0]
        return expect

    def get_entropy(self, nstep):

        if debug:
            print(inspect.stack()[0][3])

        if (not self.calc_entropy):
            raise Exception('get_entropy: calc_entropy=False. Aborting.')
        entropy = np.array([0.] * nstep)
        entropy[:] = qtf90.qutraj_run.reduced_state_entropy[:]
        return entropy

    def finalize():
        # not in use...

        if debug:
            print(inspect.stack()[0][3])

        qtf90.qutraj_run.finalize_work()
        qtf90.qutraj_run.finalize_sol()


def _gather(sols):
    # gather list of Result objects, sols, into one.
    sol = Result()
    # sol = sols[0]
    ntraj = sum([a.ntraj for a in sols])
    sol.col_times = np.zeros((ntraj), dtype=np.ndarray)
    sol.col_which = np.zeros((ntraj), dtype=np.ndarray)
    sol.col_times[0:sols[0].ntraj] = sols[0].col_times
    sol.col_which[0:sols[0].ntraj] = sols[0].col_which
    sol.states = np.array(sols[0].states)
    sol.expect = np.array(sols[0].expect)
    if (hasattr(sols[0], 'entropy')):
        sol.entropy = np.array(sols[0].entropy)
    sofar = 0
    for j in range(1, len(sols)):
        sofar = sofar + sols[j - 1].ntraj
        sol.col_times[sofar:sofar + sols[j].ntraj] = (
            sols[j].col_times)
        sol.col_which[sofar:sofar + sols[j].ntraj] = (
            sols[j].col_which)
        if (config.e_num == 0):
            if (config.options.average_states):
                # collect states, averaged over trajectories
                sol.states += np.array(sols[j].states)
            else:
                # collect states, all trajectories
                sol.states = np.vstack((sol.states,
                                        np.array(sols[j].states)))
        else:
            if (config.options.average_expect):
                # collect expectation values, averaged
                for i in range(config.e_num):
                    sol.expect[i] += np.array(sols[j].expect[i])
            else:
                # collect expectation values, all trajectories
                sol.expect = np.vstack((sol.expect,
                                        np.array(sols[j].expect)))
        if (hasattr(sols[j], 'entropy')):
            if (config.options.average_states or
                    config.options.average_expect):
                # collect entropy values, averaged
                sol.entropy += np.array(sols[j].entropy)
            else:
                # collect entropy values, all trajectories
                sol.entropy = np.vstack((sol.entropy,
                                         np.array(sols[j].entropy)))
    if (config.options.average_states or config.options.average_expect):
        if (config.e_num == 0):
            sol.states = sol.states / len(sols)
        else:
            sol.expect = list(sol.expect / len(sols))
            inds = np.where(config.e_ops_isherm)[0]
            for jj in inds:
                sol.expect[jj] = np.real(sol.expect[jj])
        if (hasattr(sols[0], 'entropy')):
            sol.entropy = sol.entropy / len(sols)

    # convert sol.expect array to list and fix dtypes of arrays
    if (not config.options.average_expect) and config.e_num != 0:
        temp = [list(sol.expect[ii]) for ii in range(ntraj)]
        for ii in range(ntraj):
            for jj in np.where(config.e_ops_isherm)[0]:
                temp[ii][jj] = np.real(temp[ii][jj])
        sol.expect = temp
    # convert to list/array to be consistent with qutip mcsolve
    sol.states = list(sol.states)
    return sol


#
# Functions to initialize the problem in fortran
#
def _init_tlist():
    Of = _realarray_to_fortran(config.tlist)
    qtf90.qutraj_run.init_tlist(Of, np.size(Of))


def _init_psi0():
    # Of = _qobj_to_fortranfull(config.psi0)
    Of = _complexarray_to_fortran(config.psi0)
    qtf90.qutraj_run.init_psi0(Of, np.size(Of))


def _init_ptrace_stuff(sel):
    psi0 = Qobj(config.psi0,
                dims=config.psi0_dims,
                shape=config.psi0_shape)
    qtf90.qutraj_run.init_ptrace_stuff(config.psi0_dims[0],
                                       np.array(sel) + 1,
                                       psi0.ptrace(sel).shape[0])


def _init_hamilt():
    # construct effective non-Hermitian Hamiltonian
    # H_eff = H - 0.5j*sum([c_ops[i].dag()*c_ops[i]
    #    for i in range(len(c_ops))])
    # Of = _qobj_to_fortrancsr(H_eff)
    # qtf90.qutraj_run.init_hamiltonian(Of[0],Of[1],
    #        Of[2],Of[3],Of[4])
    d = np.size(config.psi0)
    qtf90.qutraj_run.init_hamiltonian(
        _complexarray_to_fortran(config.h_data),
        config.h_ind + 1, config.h_ptr + 1, d, d)


def _init_c_ops():
    d = np.size(config.psi0)
    n = config.c_num
    first = True
    for i in range(n):
        # Of = _qobj_to_fortrancsr(c_ops[i])
        # qtf90.qutraj_run.init_c_ops(i+1,n,Of[0],Of[1],
        #        Of[2],Of[3],Of[4],first)
        qtf90.qutraj_run.init_c_ops(
            i + 1, n, _complexarray_to_fortran(config.c_ops_data[i]),
            config.c_ops_ind[i] + 1, config.c_ops_ptr[i] + 1, d, d, first)
        first = False


def _init_e_ops():
    d = np.size(config.psi0)
    # n = config.e_num
    n = len(config.e_ops_data)
    first = True
    for i in range(n):
        # Of = _qobj_to_fortrancsr(e_ops[i])
        # qtf90.qutraj_run.init_e_ops(i+1,n,Of[0],Of[1],
        #        Of[2],Of[3],Of[4],first)
        qtf90.qutraj_run.init_e_ops(
            i + 1, n, _complexarray_to_fortran(config.e_ops_data[i]),
            config.e_ops_ind[i] + 1, config.e_ops_ptr[i] + 1, d, d, first)
        first = False


#
# Misc. converison functions
#
def _realarray_to_fortran(a):
    datad = np.array(a, dtype=wpr)
    return datad


def _complexarray_to_fortran(a):
    datad = np.array(a, dtype=wpc)
    return datad


def _qobj_to_fortranfull(A):
    datad = np.array(A.data.toarray(), dtype=wpc)
    return datad


def _qobj_to_fortrancsr(A):
    data = np.array(A.data.data, dtype=wpc)
    indices = np.array(A.data.indices)
    indptr = np.array(A.data.indptr)
    m = A.data.shape[0]
    k = A.data.shape[1]
    return data, indices + 1, indptr + 1, m, k
