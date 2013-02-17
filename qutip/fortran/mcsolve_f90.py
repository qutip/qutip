import numpy as np
from qutip import *
import qutraj_run as qtf90
import qutip.odeconfig as odeconfig
from qutip.mcsolve import _mc_data_config
from qutip._reset import _reset_odeconfig

# Working precision
wpr = dtype(float64)
wpc = dtype(complex128)

def mcsolve_f90(H,psi0,tlist,c_ops,e_ops,ntraj=500,
        options=Odeoptions(),sparse_dms=True,serial=False,
        ptrace_sel=[],calc_entropy=False):
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
    options : Odeoptions
        Instance of ODE solver options.
    sparse_dms : boolean
        If averaged density matrices are returned, they will be stored as sparse (Compressed Row Format) matrices during computation if sparse_dms = True (default), and dense matrices otherwise. Dense matrices might be preferable for smaller systems.
    serial : boolean
        If True (default is False) the solver will not make use of the multiprocessing module, and simply run in serial.
    ptrace_sel: list
        This optional argument specifies a list of components to keep when returning a partially traced density matrix. This can be convenient for large systems where memory becomes a problem, but you are only interested in parts of the density matrix.
    calc_entropy : boolean
        If ptrace_sel is specified, calc_entropy=True will have the solver return the averaged entropy over trajectories in results.entropy. This can be interpreted as a measure of entanglement. See Phys. Rev. Lett. 93, 120408 (2004), Phys. Rev. A 86, 022310 (2012).

    Returns
    -------
    results : Odedata    
        Object storing all results from simulation.

    """
    if psi0.type!='ket':
        raise Exception("Initial state must be a state vector.")
    odeconfig.options = options
    #set num_cpus to the value given in qutip.settings 
    # if none in Odeoptions
    if not odeconfig.options.num_cpus:
        odeconfig.options.num_cpus=qutip.settings.num_cpus
    #set initial value data
    if options.tidy:
        odeconfig.psi0=psi0.tidyup(options.atol).full()
    else:
        odeconfig.psi0=psi0.full()
    odeconfig.psi0_dims=psi0.dims
    odeconfig.psi0_shape=psi0.shape
    #set general items
    odeconfig.tlist=tlist
    if isinstance(ntraj,(list,ndarray)):
        print 'mcsolve_f90: Sorry, ntraj as list argument is not supported.'
        return
        #odeconfig.ntraj=sort(ntraj)[-1]
        #ntraj_list = ntraj
    else:
        odeconfig.ntraj=ntraj
        #ntraj_list = [ntraj]
    #set norm finding constants
    odeconfig.norm_tol=options.norm_tol
    odeconfig.norm_steps=options.norm_steps

    if (not options.rhs_reuse):
        _reset_odeconfig()
        # no time dependence
        odeconfig.tflag=0
        # no gui
        odeconfig.options.gui=False
        #check for collapse operators
        if len(c_ops)>0:
            odeconfig.cflag=1
        else:
            odeconfig.cflag=0
        #Configure data
        _mc_data_config(H,psi0,[],c_ops,[],[],e_ops,options)
        # We don't use the tdfunc structure
        odeconfig.tdfunc = None

    # Load Monte Carlo class
    mc = _MC_class()
    # Set solver type
    if (options.method == 'adams'):
        mc.mf = 10
    elif (options.method == 'bdf'):
        mc.mf = 22
    else:
        print 'Unrecognized method for ode solver, using "adams".'
        mc.mf = 10
    # store ket and density matrix dims and shape for convenience
    mc.psi0_dims = psi0.dims
    mc.psi0_shape = psi0.shape
    mc.dm_dims = (psi0*psi0.dag()).dims
    mc.dm_shape = (psi0*psi0.dag()).shape
    # use sparse density matrices during computation?
    mc.sparse_dms = sparse_dms
    # run in serial?
    mc.serial_run = serial
    # are we doing a partial trace for returned states?
    mc.ptrace_sel = ptrace_sel
    if (ptrace_sel != []):
        print 'ptrace_sel set to',ptrace_sel
        print 'ps. We are using dense density matrices during computation when performing partial trace. Setting sparse_dms = False'
        print 'This feature is experimental.'
        mc.sparse_dms = False
        mc.dm_dims = psi0.ptrace(ptrace_sel).dims
        mc.dm_shape = psi0.ptrace(ptrace_sel).shape
    if (calc_entropy):
        if (ptrace_sel == []):
            print 'calc_entropy = True, but ptrace_sel = []. Please set a list of components to keep when calculating average entropy of reduced density matrix in ptrace_sel. Setting calc_entropy = False.'
            calc_entropy = False
        mc.calc_entropy = calc_entropy

    # construct output Odedata object
    output = Odedata()

    # Run
    mc.run()
    output.states = mc.sol.states
    output.expect = mc.sol.expect
    output.col_times=mc.sol.col_times
    output.col_which=mc.sol.col_which
    if (hasattr(mc.sol,'entropy')):
        output.entropy = mc.sol.entropy

    output.solver = 'Fortran 90 Monte Carlo solver'
    #simulation parameters
    output.times=odeconfig.tlist
    output.num_expect=odeconfig.e_num
    output.num_collapse=odeconfig.c_num
    output.ntraj=odeconfig.ntraj

    return output

class _MC_class():
    def __init__(self):
        self.cpus = odeconfig.options.num_cpus
        self.nprocs = self.cpus
        self.sol = Odedata()
        self.mf = 10
        # If returning density matrices, return as sparse or dense?
        self.sparse_dms = True
        # Run in serial?
        self.serial_run = False
        self.ntraj = odeconfig.ntraj
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
        self.ntrajs = []
        for i in range(self.cpus):
            self.ntrajs.append(min(int(floor(float(self.ntraj)
                /self.cpus)),
                self.ntraj-sum(self.ntrajs)))
        cnt = sum(self.ntrajs)
        while cnt<self.ntraj:
            for i in range(self.cpus):
                self.ntrajs[i] += 1
                cnt+=1
                if (cnt>=self.ntraj):
                    break
        self.ntrajs = np.array(self.ntrajs)
        self.ntrajs = self.ntrajs[np.where(self.ntrajs>0)]
        self.nprocs = len(self.ntrajs)
        sols = []
        processes = []
        resq = JoinableQueue()
        print "Number of cpus:", self.cpus
        print "Trying to start", self.nprocs, "process(es)."
        print "Number of trajectories for each process:"
        print self.ntrajs
        for i in range(self.nprocs):
            p = Process(target=self.evolve_serial,
                    args=((resq,self.ntrajs[i],i,self.seed*(i+1)),))
            p.start()
            processes.append(p)
        resq.join()
        cnt = 0
        while True:
            try:
                sols.append(resq.get())
                resq.task_done()
                cnt += 1
                if (cnt >= self.nprocs): break
            except KeyboardInterrupt:
                break
            except:
                pass
        resq.join()
        for proc in processes:
            try:
                proc.join()
            except KeyboardInterrupt:
                print("Cancel thread on keyboard interrupt")
                proc.terminate()
                proc.join()
        resq.close()
        return sols

    def serial(self):
        self.nprocs = 1
        self.ntrajs = [self.ntraj]
        print "Running in serial."
        print "Number of trajectories:", self.ntraj
        sol = self.evolve_serial((0,self.ntraj,0,self.seed))
        return [sol]

    def run(self):
        from numpy.random import random_integers
        if (odeconfig.c_num == 0):
            # force one trajectory if no collapse operators
            odeconfig.ntraj=1
            self.ntraj=1
            # Set unravel_type to 1 to integrate without collapses
            self.unravel_type = 1
            if (odeconfig.e_num==0):
                # If we are returning states, and there are no
                # collapse operators, set mc_avg to False to return
                # ket vectors instead of density matrices
                odeconfig.options.mc_avg=False
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

    def evolve_serial(self,args):
        # run ntraj trajectories for one process via fortran
        # get args
        queue,ntraj,instanceno,rngseed = args
        # initialize the problem in fortran
        _init_tlist()
        _init_psi0()
        if (self.ptrace_sel != []):
            _init_ptrace_stuff(self.ptrace_sel)
        _init_hamilt()
        if (odeconfig.c_num != 0):
            _init_c_ops()
        if (odeconfig.e_num != 0):
            _init_e_ops()
        # set options
        qtf90.qutraj_run.n_c_ops = odeconfig.c_num
        qtf90.qutraj_run.n_e_ops = odeconfig.e_num
        qtf90.qutraj_run.ntraj = ntraj
        qtf90.qutraj_run.unravel_type = self.unravel_type
        qtf90.qutraj_run.mc_avg = odeconfig.options.mc_avg
        qtf90.qutraj_run.init_odedata(odeconfig.psi0_shape[0],
                odeconfig.options.atol,odeconfig.options.rtol,mf=self.mf,
                norm_steps=odeconfig.norm_steps,norm_tol=odeconfig.norm_tol)
        # set optional arguments
        qtf90.qutraj_run.order = odeconfig.options.order
        qtf90.qutraj_run.nsteps = odeconfig.options.nsteps
        qtf90.qutraj_run.first_step = odeconfig.options.first_step
        qtf90.qutraj_run.min_step = odeconfig.options.min_step
        qtf90.qutraj_run.max_step = odeconfig.options.max_step
        qtf90.qutraj_run.norm_steps=odeconfig.options.norm_steps
        qtf90.qutraj_run.norm_tol=odeconfig.options.norm_tol
        # use sparse density matrices during computation?
        qtf90.qutraj_run.rho_return_sparse = self.sparse_dms
        # calculate entropy of reduced density matrice?
        qtf90.qutraj_run.calc_entropy = self.calc_entropy
        # run
        qtf90.qutraj_run.evolve(instanceno,rngseed)
        # construct Odedata instance
        sol = Odedata()
        sol.ntraj = ntraj
        #sol.col_times = qtf90.qutraj_run.col_times
        #sol.col_which = qtf90.qutraj_run.col_which-1
        sol.col_times, sol.col_which = self.get_collapses(ntraj)
        if (odeconfig.e_num==0):
            sol.states = self.get_states(len(odeconfig.tlist),ntraj)
        else:
            sol.expect = self.get_expect(len(odeconfig.tlist),ntraj)
        if (self.calc_entropy):
            sol.entropy = self.get_entropy(len(odeconfig.tlist))
        if (not self.serial_run):
            # put to queue
            queue.put(sol)
            #queue.put('STOP')
        #deallocate stuff
        #finalize()
        return sol

    # Routines for retrieving data data from fortran
    def get_collapses(self,ntraj):
        col_times = np.zeros((ntraj),dtype=np.ndarray)
        col_which = np.zeros((ntraj),dtype=np.ndarray)
        if (odeconfig.c_num==0):
            # no collapses
            return col_times, col_which
        for i in range(ntraj):
            qtf90.qutraj_run.get_collapses(i+1)
            times = qtf90.qutraj_run.col_times
            which = qtf90.qutraj_run.col_which
            if (times==None): times = array([])
            if (which==None): which = array([])
            else: which = which-1
            col_times[i] = np.array(times,copy=True)
            col_which[i] = np.array(which,copy=True)
        return col_times, col_which

    def get_states(self,nstep,ntraj):
        from scipy.sparse import csr_matrix
        if (odeconfig.options.mc_avg):
            states=np.array([Qobj()]*nstep)
            if (self.sparse_dms):
                # averaged sparse density matrices
                for i in range(nstep):
                    qtf90.qutraj_run.get_rho_sparse(i+1)
                    val = qtf90.qutraj_run.csr_val
                    col = qtf90.qutraj_run.csr_col-1
                    ptr = qtf90.qutraj_run.csr_ptr-1
                    m = qtf90.qutraj_run.csr_nrows
                    k = qtf90.qutraj_run.csr_ncols
                    states[i] = Qobj(csr_matrix((val,col,ptr),
                        (m,k)).toarray(),
                        dims=self.dm_dims,shape=self.dm_shape)
            else:
                # averaged dense density matrices
                for i in range(nstep):
                    states[i] = Qobj(qtf90.qutraj_run.sol[0,i,:,:],
                        dims=self.dm_dims,shape=self.dm_shape)
        else:
            # all trajectories as kets
            if (ntraj==1):
                states=np.array([Qobj()]*nstep)
                for i in range(nstep):
                    states[i] = Qobj(matrix(
                        qtf90.qutraj_run.sol[0,0,i,:]).transpose(),
                        dims=self.psi0_dims,shape=self.psi0_shape)
            else:
                states=np.array([np.array([Qobj()]*nstep)]*ntraj)
                for traj in range(ntraj):
                    for i in range(nstep):
                        states[traj][i] = Qobj(matrix(
                            qtf90.qutraj_run.sol[0,traj,i,:]).transpose(),
                            dims=self.psi0_dims,shape=self.psi0_shape)
        return states

    def get_expect(self,nstep,ntraj):
        if (odeconfig.options.mc_avg):
            expect=np.array([np.array([0.+0.j]*nstep)]*odeconfig.e_num)
            for j in range(odeconfig.e_num):
                expect[j] = qtf90.qutraj_run.sol[j,0,:,0]
        else:
            expect=np.array([[np.array([0.+0.j]*nstep)]*odeconfig.e_num]
                    *ntraj)
            for j in range(odeconfig.e_num):
                expect[:,j,:] = qtf90.qutraj_run.sol[j,:,:,0]
        return expect

    def get_entropy(self,nstep):
        if (not self.calc_entropy):
            print 'get_entropy: calc_entropy=False. Aborting.'
            return
        entropy = np.array([0.]*nstep)
        entropy[:] = qtf90.qutraj_run.reduced_state_entropy[:]
        return entropy

    def finalize():
        # not in use...
        qtf90.qutraj_run.finalize_work()
        qtf90.qutraj_run.finalize_sol()

def _gather(sols):
    # gather list of Odedata objects, sols, into one.
    sol = Odedata()
    #sol = sols[0]
    ntraj = sum([a.ntraj for a in sols])
    sol.col_times = np.zeros((ntraj),dtype=np.ndarray)
    sol.col_which = np.zeros((ntraj),dtype=np.ndarray)
    sol.col_times[0:sols[0].ntraj] = sols[0].col_times
    sol.col_which[0:sols[0].ntraj] = sols[0].col_which
    sol.states = np.array(sols[0].states)
    sol.expect = np.array(sols[0].expect)
    if (hasattr(sols[0],'entropy')):
        sol.entropy = np.array(sols[0].entropy)
    sofar = 0
    for j in range(1,len(sols)):
        sofar = sofar + sols[j-1].ntraj
        sol.col_times[sofar:sofar+sols[j].ntraj] = (
                sols[j].col_times)
        sol.col_which[sofar:sofar+sols[j].ntraj] = (
                sols[j].col_which)
        if (odeconfig.e_num==0):
            if (odeconfig.options.mc_avg):
                # collect states, averaged over trajectories
                sol.states += np.array(sols[j].states)
            else:
                # collect states, all trajectories
                sol.states = np.vstack((sol.states,
                    np.array(sols[j].states)))
        else:
            if (odeconfig.options.mc_avg):
                # collect expectation values, averaged
                for i in range(odeconfig.e_num):
                    sol.expect[i] += np.array(sols[j].expect[i])
            else:
                # collect expectation values, all trajectories
                sol.expect = np.vstack((sol.expect,
                    np.array(sols[j].expect)))
        if (hasattr(sols[j],'entropy')):
            if (odeconfig.options.mc_avg):
                # collect entropy values, averaged
                sol.entropy += np.array(sols[j].entropy)
            else:
                # collect entropy values, all trajectories
                sol.entropy = np.vstack((sol.entropy,
                    np.array(sols[j].entropy)))
    if (odeconfig.options.mc_avg):
        if (odeconfig.e_num==0):
            sol.states = sol.states/len(sols)
        else:
            sol.expect = sol.expect/len(sols)
        if (hasattr(sols[0],'entropy')):
            sol.entropy = sol.entropy/len(sols)
    # convert to list/array to be consistent with qutip mcsolve
    sol.states = list(sol.states)
    #sol.expect = list(sol.expect)
    return sol

#
# Functions to initialize the problem in fortran
#

def _init_tlist():
    Of = _realarray_to_fortran(odeconfig.tlist)
    qtf90.qutraj_run.init_tlist(Of,
            size(Of))

def _init_psi0():
    #Of = _qobj_to_fortranfull(odeconfig.psi0)
    Of = _complexarray_to_fortran(odeconfig.psi0)
    qtf90.qutraj_run.init_psi0(Of,size(Of))

def _init_ptrace_stuff(sel):
    psi0 = Qobj(odeconfig.psi0,
            dims=odeconfig.psi0_dims,shape=odeconfig.psi0_shape)
    qtf90.qutraj_run.init_ptrace_stuff(odeconfig.psi0_dims[0],
            np.array(sel)+1,psi0.ptrace(sel).shape[0])

def _init_hamilt():
    # construct effective non-Hermitian Hamiltonian
    #H_eff = H - 0.5j*sum([c_ops[i].dag()*c_ops[i]
    #    for i in range(len(c_ops))])
    #Of = _qobj_to_fortrancsr(H_eff)
    #qtf90.qutraj_run.init_hamiltonian(Of[0],Of[1],
    #        Of[2],Of[3],Of[4])
    d = size(odeconfig.psi0)
    qtf90.qutraj_run.init_hamiltonian(
            _complexarray_to_fortran(odeconfig.h_data),
            odeconfig.h_ind+1,odeconfig.h_ptr+1,d,d)

def _init_c_ops():
    d = size(odeconfig.psi0)
    n = odeconfig.c_num
    first = True
    for i in range(n):
        #Of = _qobj_to_fortrancsr(c_ops[i])
        #qtf90.qutraj_run.init_c_ops(i+1,n,Of[0],Of[1],
        #        Of[2],Of[3],Of[4],first)
        qtf90.qutraj_run.init_c_ops(i+1,n,
                _complexarray_to_fortran(odeconfig.c_ops_data[i]),
                odeconfig.c_ops_ind[i]+1,odeconfig.c_ops_ptr[i]+1,d,d,
                first)
        first = False

def _init_e_ops():
    d = size(odeconfig.psi0)
    #n = odeconfig.e_num
    n = len(odeconfig.e_ops_data)
    first = True
    for i in range(n):
        #Of = _qobj_to_fortrancsr(e_ops[i])
        #qtf90.qutraj_run.init_e_ops(i+1,n,Of[0],Of[1],
        #        Of[2],Of[3],Of[4],first)
        qtf90.qutraj_run.init_e_ops(i+1,n,
                _complexarray_to_fortran(odeconfig.e_ops_data[i]),
                odeconfig.e_ops_ind[i]+1,odeconfig.e_ops_ptr[i]+1,d,d,
                first)
        first = False
#
# Misc. converison functions
#

def _realarray_to_fortran(a):
    datad = np.array(a,dtype=wpr)
    return datad

def _complexarray_to_fortran(a):
    datad = np.array(a,dtype=wpc)
    return datad

def _qobj_to_fortranfull(A):
    datad = np.array(A.data.toarray(),dtype=wpc)
    return datad

def _qobj_to_fortrancsr(A):
    data = np.array(A.data.data,dtype=wpc)
    indices = np.array(A.data.indices)
    indptr = np.array(A.data.indptr)
    m = A.data.shape[0]
    k = A.data.shape[1]
    return data,indices+1,indptr+1,m,k

