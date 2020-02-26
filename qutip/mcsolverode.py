from numpy.random import RandomState, randint
import numpy as np
from numpy.linalg import norm as la_norm
from scipy.integrate import solve_ivp, ode
from qutip.solver import ExpectOps
from qutip.cy.spmatfuncs import normalize_inplace, normalize_op_inplace
from qutip.parallel import parallel_map, serial_map
from .qobjevo import QobjEvo
from .qobj import Qobj
from qutip.cy.mcsolve import CyMcOde, CyMcOdeDiag
from scipy.integrate._ode import zvode

class qutip_zvode(zvode):
    def step(self, *args):
        itask = self.call_args[2]
        self.rwork[0] = args[4]
        self.call_args[2] = 5
        r = self.run(*args)
        self.call_args[2] = itask
        return r


def normalize_prop(state):
    st = st.resize(())
    norms = la_norm(state, axis=0)
    state /= norms
    return np.mean(norms)


def dummy_normalize(state):
    return 0


normalize_dm = dummy_normalize


class McOdeSolver:
    def __init__(self, LH, options, progress_bar):
        self.LH = LH
        self.options = options
        self.progress_bar = progress_bar
        self.statetype = "dense"

    def run(self, state0, tlist, args={}):
        raise NotImplementedError

    def step(self, state, t_in, t_out):
        raise NotImplementedError

    def update_args(self, args):
        self.LH.arguments(args)
        [c.arguments(args) for c in self.td_c_ops]
        [c.arguments(args) for c in self.td_n_ops]

    @staticmethod
    def _prepare_e_ops(e_ops):
        if isinstance(e_ops, ExpectOps):
            return e_ops
        else:
            return ExpectOps(e_ops)

    def _prepare_normalize_func(self, state0):
        opt = self.options
        size = np.prod(state0.shape)
        if opt.normalize_output and size == self.LH.shape[1]:
            if self.LH.cte.issuper:
                print("normalize_dm")
                self.normalize_func = normalize_dm
            else:
                print("normalize_inplace")
                self.normalize_func = normalize_inplace
        elif opt.normalize_output and size == np.prod(self.LH.shape):
            print("normalize_prop")
            self.normalize_func = normalize_op_inplace
        elif opt.normalize_output:
            print("normalize_mixed", size, self.LH.shape)
            self.normalize_func = normalize_mixed(state0.shape)

    def step(self, t, reset=False, changed=False):
        raise NotImplementedError("Stepper not available for mcsolver")


class McOdeScipyZvode(McOdeSolver):
    def __init__(self, LH, c_ops, options, parallel, progress_bar):
        self.LH = LH
        self.c_ops = c_ops
        self._make_system(LH, c_ops)
        self.options = options
        self.progress_bar = progress_bar
        self.map_func = parallel_map if parallel else serial_map
        self.statetype = "dense"
        self.name = "scipy_zvode_mc"
        self.normalize_func = dummy_normalize

    def _make_system(self, H, c_ops):
        self.td_c_ops = []
        self.td_n_ops = []
        self.Hevo = -1j * H
        for c in c_ops:
            cevo = c
            cdc = cevo._cdc()
            self.Hevo += -0.5 * cdc
            cevo.compile()
            cdc.compile()
            self.td_c_ops.append(cevo)
            self.td_n_ops.append(cdc)
        self.Hevo.compile()

    def run(self, state0, num_traj, seeds, tlist, args={}, e_ops=[]):
        """
        Internal function for solving ODEs.
        """
        if args: self.update_args(args)
        self.e_ops = self._prepare_e_ops(e_ops)
        self.state0 = state0
        self.tlist = tlist
        n_tsteps = len(tlist)
        state_size = np.prod(state0.shape)
        num_saved_state = n_tsteps if self.options.store_states else 1
        states = np.zeros((num_saved_state, state_size), dtype=complex)

        map_func = self.map_func
        if self.options.num_cpus == 1 or num_traj == 1:
            map_func = serial_map
        map_kwargs = {'progress_bar': self.progress_bar,
                      'num_cpus': self.options.num_cpus}

        results = map_func(self._single_traj, seeds, **map_kwargs)

        return results

    def set(self, state0, t0):
        opt = self.options
        func = self.Hevo._get_mul(state0)
        r = ode(func)
        options_keys = ['atol', 'rtol', 'nsteps', 'method', 'order'
                        'first_step', 'max_step','min_step']
        options = {key:getattr(opt, key)
                   for key in options_keys
                   if hasattr(opt, key)}
        r.set_integrator('zvode', method="adams")
        r._integrator = qutip_zvode(**options)
        if isinstance(state0, Qobj):
            initial_vector = state0.full().ravel('F')
        else:
            initial_vector = state0
        r.set_initial_value(initial_vector, t0)
        return r
        #self._prepare_normalize_func(state0)

    def _single_traj(self, seed):
        """
        Monte Carlo algorithm returning state-vector or expectation values
        at times tlist for a single trajectory.
        """
        # SEED AND RNG AND GENERATE
        prng = RandomState(seed)

        # set initial conditions
        tlist = self.tlist
        e_ops = self.e_ops.copy()
        e_ops.init(tlist)

        ODE = self.set(self.state0, self.tlist[0])
        cymc = CyMcOde(self.Hevo, self.td_c_ops, self.td_n_ops, self.options)

        states_out, ss_out, collapses = cymc.run_ode(ODE, self.tlist,
                                                     e_ops, prng)

        # Run at end of mc_alg function
        # -----------------------------
        if self.options.steady_state_average:
            ss_out /= float(len(tlist))

        return (states_out, ss_out, e_ops.raw_out, collapses)

"""
class McOdeQutipDiag(McOdeSolver):
    def __init__(self, LH, options, progress_bar):
        self.LH = LH
        self.options = options
        self.progress_bar = progress_bar
        self.statetype = "dense"
        self.name = "qutip_diag_mc"
        self.normalize_func = dummy_normalize

    @staticmethod
    def funcwithfloat(func):
        def new_func(t, y):
            y_cplx = y.view(complex)
            dy = func(t, y_cplx)
            return dy.view(np.float64)
        return new_func

    def run(self, state0, tlist, args={}, e_ops=[]):
        "#""
        Internal function for solving ODEs.
        "#""
        #TODO: normalization in solver
        # > v1: event, step when norm bad
        # > v2: extra non-hermitian term to
        opt = self.options
        normalize_func = self.normalize_func
        e_ops = self._prepare_e_ops(e_ops)
        self.LH.arguments(args)
        n_tsteps = len(tlist)
        state_size = np.prod(state0.shape)
        num_saved_state = n_tsteps if opt.store_states else 1
        states = np.zeros((num_saved_state, state_size), dtype=complex)
        e_ops_store = bool(e_ops)
        e_ops.init(tlist)

        self.set(state0, tlist[0])
        ode_res = solve_ivp(self.func, [tlist[0], tlist[-1]],
                            self._y, t_eval=tlist, **self.ivp_opt)

        e_ops.init(tlist)
        for t_idx, cdata in enumerate(ode_res.y.T):
            y_cplx = cdata.copy().view(complex)
            self.normalize_func(y_cplx)
            if opt.store_states:
                states[t_idx, :] = y_cplx
            e_ops.step(t_idx, y_cplx)
        if not opt.store_states:
            states[0, :] = cdata
        return states, e_ops.finish()

    def step(self, t, reset=False, changed=False):
        ode_res = solve_ivp(self.func, [self._t, t], self._y,
                            t_eval=[t], **self.ivp_opt)
        self._y = ode_res.y.T[0].view(complex)
        self._t = t
        self.normalize_func(self._y)
        return self._y

    def set(self, state0, t0):
        opt = self.options
        self._t = t0
        self.func = self.funcwithfloat(self.LH._get_mul(state0))
        if isinstance(state0, Qobj):
            self._y = state0.full().ravel('F').view(np.float64)
        else:
            self._y = state0.view(np.float64)

        options_keys = ['method', 'atol', 'rtol',
                        'nsteps']
        self.ivp_opt = {key:getattr(opt, key)
                        for key in options_keys
                        if hasattr(opt, key)}
        self._prepare_normalize_func(state0)

    def make_diag_system(self, H, c_ops):
        ss = SolverSystem()
        ss.td_c_ops = []
        ss.td_n_ops = []

        H_ = H.copy()
        H_ *= -1j
        for c in c_ops:
            H_ += -0.5 * c.dag() * c

        w, v = np.linalg.eig(H_.full())
        arg = np.argsort(np.abs(w))
        eig = w[arg]
        U = v.T[arg].T
        Ud = U.T.conj()

        for c in c_ops:
            c_diag = Qobj(Ud @ c.full() @ U, dims=c.dims)
            cevo = QobjEvo(c_diag)
            cdc = cevo._cdc()
            cevo.compile()
            cdc.compile()
            ss.td_c_ops.append(cevo)
            ss.td_n_ops.append(cdc)

        ss.H_diag = eig
        ss.Ud = Ud
        ss.U = U
        ss.args = {}
        ss.type = "Diagonal"
        solver_safe["mcsolve"] = ss

        if self.e_ops and not self.e_ops.isfunc:
            e_op = [Qobj(Ud @ e.full() @ U, dims=e.dims) for e in self.e_ops.e_ops]
            self.e_ops = ExpectOps(e_ops)
        self.ss = ss
        self.reset()

    def _single_traj_diag(self, nt):
        "#""
        Monte Carlo algorithm returning state-vector or expectation values
        at times tlist for a single trajectory.
        "#""
        # SEED AND RNG AND GENERATE
        prng = RandomState(self.seeds[nt])
        opt = self.options

        ss = self.ss
        tlist = self.tlist
        e_ops = self.e_ops.copy()
        opt = self.options
        e_ops.init(tlist)

        cymc = CyMcOdeDiag(ss, opt)
        states_out, ss_out, collapses = cymc.run_ode(self.initial_vector, tlist,
                                                     e_ops, prng)

        if opt.steady_state_average:
            ss_out = ss.U @ ss_out @ ss.Ud
        states_out = np.inner(ss.U, states_out).T
        if opt.steady_state_average:
            ss_out /= float(len(tlist))
        return (states_out, ss_out, e_ops, collapses)


class McOdeScipyIVP(McOdeSolver):
    def __init__(self, H, c_ops, tlist, args,
                 options, progress_bar):
        self.H = H
        self.c_ops = c_ops
        self.options = options
        self.progress_bar = progress_bar
        self.statetype = "dense"
        self.name = "scipy_ivp_mc"
        self.normalize_func = dummy_normalize

    @staticmethod
    def funcwithfloat(func):
        def new_func(t, y):
            y_cplx = y.view(complex)
            dy = func(t, y_cplx)
            return dy.view(np.float64)
        return new_func

    def _make_system(self, H, c_ops, tlist=None, args={}):
        col_args = _collapse_args(args)
        self.td_c_ops = []
        self.td_n_ops = []
        self.Hevo = qobjevo_maker(H, args, tlist)
        self.Hevo *= -1j
        for c in c_ops:
            cevo = qobjevo_maker(c, args, tlist)
            cdc = cevo._cdc()
            self.Hevo += -0.5 * cdc
            cevo.compile()
            cdc.compile()
            self.td_c_ops.append(cevo)
            self.td_n_ops.append(cdc)
        self.Hevo.compile()

    def run(self, state0, tlist, args={}, e_ops=[]):
        "#""
        Internal function for solving ODEs.
        "#""
        #TODO: normalization in solver
        # > v1: event, step when norm bad
        # > v2: extra non-hermitian term to
        opt = self.options
        normalize_func = self.normalize_func
        e_ops = self._prepare_e_ops(e_ops)
        self.LH.arguments(args)
        n_tsteps = len(tlist)
        state_size = np.prod(state0.shape)
        num_saved_state = n_tsteps if opt.store_states else 1
        states = np.zeros((num_saved_state, state_size), dtype=complex)
        e_ops_store = bool(e_ops)
        e_ops.init(tlist)

        self.set(state0, tlist[0])
        ode_res = solve_ivp(self.func, [tlist[0], tlist[-1]],
                            self._y, t_eval=tlist, **self.ivp_opt)

        e_ops.init(tlist)
        for t_idx, cdata in enumerate(ode_res.y.T):
            y_cplx = cdata.copy().view(complex)
            self.normalize_func(y_cplx)
            if opt.store_states:
                states[t_idx, :] = y_cplx
            e_ops.step(t_idx, y_cplx)
        if not opt.store_states:
            states[0, :] = cdata
        return states, e_ops.finish()

    def step(self, t, reset=False, changed=False):
        ode_res = solve_ivp(self.func, [self._t, t], self._y,
                            t_eval=[t], **self.ivp_opt)
        self._y = ode_res.y.T[0].view(complex)
        self._t = t
        self.normalize_func(self._y)
        return self._y

    def set(self, state0, t0):
        opt = self.options
        self._t = t0
        self.func = self.funcwithfloat(self.LH._get_mul(state0))
        if isinstance(state0, Qobj):
            self._y = state0.full().ravel('F').view(np.float64)
        else:
            self._y = state0.view(np.float64)

        options_keys = ['method', 'atol', 'rtol',
                        'nsteps']
        self.ivp_opt = {key:getattr(opt, key)
                        for key in options_keys
                        if hasattr(opt, key)}
        self._prepare_normalize_func(state0)
"""
