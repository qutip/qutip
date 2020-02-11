


import numpy as np
from scipy.integrate import solve_ivp, ode
from qutip.qobjevo_maker import is_dynargs_pattern
from qutip.cy.spmatfuncs import normalize_inplace

normalize_dm = None

def normalize_prop(state):
    norms = la_norm(state, axis=0)
    state /= norms
    return np.mean(norms)


def dummy_normalize(state):
    return 0


def stack_rho(rhos):
    size = rhos[0].shape[0] * rhos[0].shape[1]
    out = np.zeros((size, len(rhos)), dtype=complex)
    for i, rho in enumerate(rhos):
        out[:,i] = rho.full().ravel("F")
    return [Qobj(out)]


def _islistof(obj, type_, default, errmsg):
    if isinstance(obj, type_):
        obj = [obj]
        n = 0
    elif isinstance(obj, list):
        if any((not isinstance(ele, type_) for ele in obj)):
            raise TypeError(errmsg)
        n = len(obj)
    elif not obj and default is not None:
        obj = default
        n = len(obj)
    else:
        raise TypeError(errmsg)
    return obj, n


def stack_ket(kets):
    # TODO: speedup, Qobj to dense to Qobj, probably slow
    out = np.zeros((kets[0].shape[0], len(kets)), dtype=complex)
    for i, ket in enumerate(kets):
        out[:,i] = ket.full().ravel()
    return [Qobj(out)]


class OdeSolver:
    """Parent of OdeSolver used by Qutip quantum system solvers.
    Do not use directly, but use child class.

    Parameters
    ----------
    data : array_like
        Sparse matrix characterizing the quantum object.


    Attributes
    ----------
    data : array_like
        Sparse matrix characterizing the quantum object.


    Methods
    -------
    run(state0, tlist, )
        Create copy of Qobj

    Child
    -----
    OdeScipyZvode

    OdeScipyDop853

    OdeScipyIVP

    Futur:
        ?OdeQutipDopri:
        ?OdeQutipAdam:
        ?OdeDiagonalized:
        ?OdeSparse:
        ?OdeAdaptativeHilbertSpace:

    """
    def __init__(self, LH, options, progress_bar):
        self.LH = LH
        self.options = options
        self.progress_bar = progress_bar
        self.statetype = "dense"

    def run(self, state0, tlist, args={}):
        raise NotImplementedError

    def step(self, state, t_in, t_out):
        raise NotImplementedError

    def prepare(self):
        pass

    def update_args(self, args):
        self.LH.arguments(args)

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
                self.normalize_func = normalize_dm
            else:
                self.normalize_func = normalize_inplace
        elif opt.normalize_output and size == np.prod(self.LH.shape):
            self.normalize_func = normalize_prop
        elif opt.normalize_output:
            self.normalize_func = normalize_mixed(state0.shape)


class OdeScipyZvode(OdeSolver):
    # -------------------------------------------------------------------------
    # Solve an ODE for func.
    # Calculate the required expectation values or invoke callback
    # function at each time step.
    def __init__(self, LH, options, progress_bar):
        self.LH = LH
        self.options = options
        self.progress_bar = progress_bar
        self.statetype = "dense"
        self.name = "scipy_zvode"
        self._r = None
        self.normalize_func = dummy_normalize
        self._error_msg = ("ODE integration error: Try to increase "
                           "the allowed number of substeps by increasing "
                           "the nsteps parameter in the Options class.")

    def run(self, state0, tlist, args={}, e_ops=[]):
        """
        Internal function for solving ODEs.
        """
        opt = self.options
        normalize_func = self.normalize_func
        self.LH.arguments(args)
        e_ops = self._prepare_e_ops(e_ops)
        n_tsteps = len(tlist)
        state_size = np.prod(state0.shape)
        num_saved_state = n_tsteps if opt.store_states else 1

        states = np.zeros((num_saved_state, state_size), dtype=complex)
        e_ops_store = bool(e_ops)
        e_ops.init(tlist)

        self.set(state0, tlist[0])
        r = self._r

        self.progress_bar.start(n_tsteps-1)
        for t_idx, t in enumerate(tlist):
            if not r.successful():
                raise Exception(self._error_msg)
            # get the current state / oper data if needed
            if opt.store_states or opt.normalize_output or e_ops_store:
                cdata = r._y
                if self.normalize_func(cdata) > opt.atol:
                    r.set_initial_value(cdata, r.t)
                if opt.store_states:
                    states[t_idx, :] = cdata
                e_ops.step(t_idx, cdata)
            if t_idx < n_tsteps - 1:
                r.integrate(tlist[t_idx+1])
                self.progress_bar.update(t_idx)
        self.progress_bar.finished()
        states[-1, :] = r.y
        self.normalize_func(states[-1, :])
        return states, e_ops.finish()

    def step(self, t, reset=False, changed=False):
        if changed or reset:
            self.set(self._r.y, self._r.t)
        self._r.integrate(t)
        state = self._r.y
        self.normalize_func(state)
        return state

    def set(self, state0, t0):
        opt = self.options
        func = self.LH._get_mul(state0)
        r = ode(func)
        options_keys = ['atol', 'rtol', 'nsteps', 'method', 'order'
                        'first_step', 'max_step','min_step']
        options = {key:getattr(opt, key)
                   for key in options_keys
                   if hasattr(opt, key)}
        r.set_integrator('zvode', **options)
        if isinstance(state0, Qobj):
            initial_vector = state0.full().ravel('F')
        else:
            initial_vector = state0
        r.set_initial_value(initial_vector, t0)
        self._r = r
        self._prepare_normalize_func(state0)


class OdeScipyDop853(OdeSolver):
    # -------------------------------------------------------------------------
    # Solve an ODE for func.
    # Calculate the required expectation values or invoke callback
    # function at each time step.
    def __init__(self, LH, options, progress_bar):
        self.LH = LH
        self.options = options
        self.progress_bar = progress_bar
        self.statetype = "dense"
        self.name = "scipy_dop853"
        self._r = None
        self.normalize_func = dummy_normalize
        self._error_msg = ("ODE integration error: Try to increase "
                           "the allowed number of substeps by increasing "
                           "the nsteps parameter in the Options class.")

    def run(self, state0, tlist, args={}, e_ops=[]):
        """
        Internal function for solving ODEs.
        """
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
        r = self._r

        self.progress_bar.start(n_tsteps-1)
        for t_idx, t in enumerate(tlist):
            if not r.successful():
                raise Exception(self._error_msg)
            # get the current state / oper data if needed
            if opt.store_states or opt.normalize_output or e_ops_store:
                cdata = r.y.view(complex)
                if self.normalize_func(cdata) > opt.atol:
                    r.set_initial_value(cdata.view(np.float64), r.t)
                if opt.store_states:
                    states[t_idx, :] = cdata
                e_ops.step(t_idx, cdata)
            if t_idx < n_tsteps - 1:
                r.integrate(tlist[t_idx+1])
                self.progress_bar.update(t_idx)
        self.progress_bar.finished()
        states[-1, :] = r.y.view(complex)
        self.normalize_func(states[-1,:])
        return states, e_ops.finish()

    @staticmethod
    def funcwithfloat(func):
        def new_func(t, y):
            y_cplx = y.view(complex)
            dy = func(t, y_cplx)
            return dy.view(np.float64)
        return new_func

    def step(self, t, reset=False, changed=False):
        if reset:
            self.set(self._r.y.view(complex), self._r.t)
        self._r.integrate(t)
        state = self._r.y.view(complex)
        self.normalize_func(state)
        return state

    def set(self, state0, t0):
        opt = self.options
        func = self.LH._get_mul(state0)
        r = ode(self.funcwithfloat(func))
        options_keys = ['atol', 'rtol', 'nsteps', 'first_step', 'max_step',
                        'ifactor', 'dfactor', 'beta']
        options = {key:getattr(opt, key)
                   for key in options_keys
                   if hasattr(opt, key)}
        r.set_integrator('dop853', **options)
        if isinstance(state0, Qobj):
            initial_vector = state0.full().ravel('F').view(np.float64)
        else:
            initial_vector = state0.view(np.float64)
        r.set_initial_value(initial_vector, t0)
        self._r = r
        self._prepare_normalize_func(state0)


class OdeScipyIVP(OdeSolver):
    def __init__(self, LH, options, progress_bar):
        self.LH = LH
        self.options = options
        self.progress_bar = progress_bar
        self.statetype = "dense"
        self.name = "scipy_ivp"
        self.normalize_func = dummy_normalize

    @staticmethod
    def funcwithfloat(func):
        def new_func(t, y):
            y_cplx = y.view(complex)
            dy = func(t, y_cplx)
            return dy.view(np.float64)
        return new_func

    def run(self, state0, tlist, args={}, e_ops=[]):
        """
        Internal function for solving ODEs.
        """
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


class MESolver(Solver):
    """Master Equation Solver

    """
    def __init__(self, H, c_ops=[], args={}, tlist=[], options=None):
        if options is None:
            options = Options()

        super().__init__()
        if isinstance(H, (list, Qobj, QobjEvo)):
            ss = _mesolve_QobjEvo(H, c_ops, tlist, args, options)
        elif callable(H):
            ss = _mesolve_func_td(H, c_ops, args, options)
        else:
            raise Exception("Invalid H type")

        self.H = H
        self.ss = ss
        self.c_ops = []
        self.dims = None
        self._args = args
        self.tlist = tlist
        self.options = options
        self._optimization = {"period":0}

    def set_initial_value(self, rho0, tlist=[]):
        self.state0 = rho0
        self.dims = rho0.dims
        if tlist:
            self.tlist = tlist

    def optimization(self, period=0, sparse=False):
        self._optimization["period"] = period
        self._optimization["sparse"] = sparse
        raise NotImplementedError

    def run(self, progress_bar=True):
        if progress_bar is True:
            progress_bar = TextProgressBar()

        func, ode_args = self.ss.makefunc(self.ss, self.state0,
                                          self._args, self.options)
        old_store_state = self._options.store_states
        if not self.e_ops:
            self._options.store_states = True

        func, ode_args = self.ss.makeoper(self.ss, self.state0,
                                          self._args, self.options)

        if not self._options.normalize_output:
            normalize_func = None

        self._e_ops.init(self._tlist)
        self._state_out = self._generic_ode_solve(func, ode_args, self.state0,
                                                  self._tlist, self._e_ops,
                                                  False, self._options,
                                                  progress_bar)
        self._options.store_states = old_store_state

    def batch_run(self, states=[], args_sets=[],
                  progress_bar=True, map_func=parallel_map):
        N_states0 = len(states)
        N_args = len(args_sets)

        if not states:
            states = [self.state0]
        states = [ket2dm(state) if isket(state) else state for state in states]
        size = rhos[0].shape[0] * rhos[0].shape[1]

        if not args_sets:
            args_sets = [self._args]

        if progress_bar is True:
            progress_bar = TextProgressBar()
        map_kwargs = {'progress_bar': progress_bar,
                      'num_cpus': self.options.num_cpus}

        if self.ss.with_state:
            state, expect = self._batch_run_rho(states, args_sets,
                                                map_func, map_kwargs)
        elif N_states0 > size:
            state, expect = self._batch_run_prop_rho(states, args_sets,
                                                     map_func, map_kwargs)
        elif N_states0 >= 2:
            state, expect = self._batch_run_merged_rho(states, args_sets,
                                                       map_func, map_kwargs)
        else:
            state, expect = self._batch_run_rho(states, args_sets,
                                                map_func, map_kwargs)

        states_out = np.empty((num_states, num_args, nt), dtype=object)
        for i,j,k in product(range(num_states), range(num_args), range(nt)):
            oper = state[i,j,k].reshape((vec_len, vec_len), order="F")
            states_out[i,j,k] = dense2D_to_fastcsr_fmode(oper, vec_len, vec_len)
        return states_out, expect

    def _batch_run_rho(self, states, args_sets, map_func, map_kwargs):
        N_states0 = len(kets)
        N_args = len(args_sets)
        nt = len(self._tlist)
        size = states[0].shape[0] * states[0].shape[1]

        states_out = np.empty((N_states0, N_args, nt, size), dtype=complex)
        expect_out = np.empty((N_states0, N_args), dtype=object)
        old_store_state = self._options.store_states
        store_states = not bool(self._e_ops) or self._options.store_states
        self._options.store_states = store_states

        values = list(product(states, args_sets))

        results = map_func(self._one_run_ket, values, (), **map_kwargs)

        for i, (state, expect) in enumerate(results):
            args_n, state_n = divmod(i, N_states0)
            if self._e_ops:
                expect_out[state_n, args_n] = expect.finish()
            if store_states:
                states_out[state_n, args_n] = state

        self._options.store_states = old_store_state
        return states_out, expect_out

    def _batch_run_prop_rho(self, states, args_sets, map_func, map_kwargs):
        N_states0 = len(states)
        N_args = len(args_sets)
        nt = len(self._tlist)
        size = states[0].shape[0] * states[0].shape[1]

        states_out = np.empty((N_states0, N_args, nt, size), dtype=complex)
        expect_out = np.empty((N_states0, N_args), dtype=object)
        old_store_state = self._options.store_states
        store_states = (not bool(self._e_ops) or self._options.store_states)
        self._options.store_states = True

        computed_state = [qeye(size)]
        values = list(product(computed_state, args_sets))

        if len(values) == 1:
            map_func = serial_map
        results = map_func(self._one_run_ket, values, (),
                           **map_kwargs)

        for args_n, (prop, _) in enumerate(results):
            for state_n, rho in enumerate(states):
                e_op = self._e_ops.copy()
                e_op.init(self._tlist)
                state = np.zeros((nt, size), dtype=complex)
                rho_vec = rho.full.ravel("F")
                for t in self._tlist:
                    state[t,:] = prop[t,:,:] @ rho_vec
                    e_op.step(t, state[t,:])
                if self._e_ops:
                    expect_out[state_n, args_n] = e_op.finish()
                if store_states:
                    states_out[state_n, args_n] = state
        self._options.store_states = old_store_state
        return states_out, expect_out

    def _batch_run_merged_rho(self, states, args_sets, map_func, map_kwargs):
        nt = len(self._tlist)
        num_states0 = len(kets)
        num_args = len(args_sets)
        size = states[0].shape[0] * states[0].shape[1]
        size_s = states[0].shape[0]

        states_out = np.empty((N_states0, N_args, nt, size), dtype=complex)
        expect_out = np.empty((num_states0, num_args), dtype=object)
        old_store_state = self._options.store_states
        store_states = (not bool(self._e_ops) or self._options.store_states)
        self._options.store_states = True
        values = list(product(stack_rho(kets), args_sets))

        if len(values) == 1:
            map_func = serial_map
        results = map_func(self._one_run_rho, values, (), **map_kwargs)

        for args_n, (state, _) in enumerate(results):
            e_ops_ = [self._e_ops.copy() for _ in range(num_states0)]
            [e_op.init(self._tlist) for e_op in e_ops_]
            states_out_run = [np.zeros((nt, size), dtype=complex)
                              for _ in range(num_states0)]
            for t in range(nt):
                state_t = state[t,:].reshape((num_states0, size)).T
                for j in range(num_states0):
                    vec = state_t[:,j]
                    e_ops_[j].step(t, vec)
                    if store_states:
                        states_out_run[j][t,:] = vec

            for state_n in range(num_states0):
                expect_out[state_n, args_n] = e_ops_[state_n].finish()
                if store_states:
                    states_out[state_n, args_n] = states_out_run[state_n]
        self._options.store_states = old_store_state
        return states_out, expect_out

    def _one_run_rho(self, run_data):
        opt = self._options
        state0, args = run_data
        func, ode_args = self.ss.makefunc(self.ss, state0, args, opt)

        if state0.isket:
            e_ops = self._e_ops.copy()
        else:
            e_ops = ExpectOps([])

        state = self._generic_ode_solve(func, ode_args, state0, self._tlist,
                                        e_ops, False, opt, BaseProgressBar())
        return state, e_ops

#


class _SolverCacheOneEvo:
    def __init__(self):
        self.times = np.array([], dtype=np.double)
        self.vals = np.array([], dtype=object)
        self.N = 0
        self.start = 0
        self.stop = 0
        self.step = 0
        self.interpolation = False

    def __setitem__(self, time_info, vals):
        if not isinstance(vals, (list, np.ndarray)):
            vals = [vals]
        vals = np.array(vals, dtype=object)
        times = self._time2arr_set(time_info, len(vals))
        if self.times.size == 0:
            self.times = times
            self.vals = np.array(vals, dtype=object)
        else:
            self.times = np.concatenate((self.times, times), axis=None)
            self.vals = np.concatenate((self.vals, vals), axis=None)
        if len(self.vals) != len(self.times):
            print(len(self.vals), len(self.times))
            print(self.start, self.stop, self.step, self.N)
            raise Exception("Error in caching")

        self._sort()
        self._remove_donblons()
        self.N = len(self.times)
        self.start, self.stop, self.step = self._to_slice(self.times)

    def __getitem__(self, time_info):
        if self.N == 0:
            return (np.array([], dtype=object),
                    np.array([], dtype=float), time_info)
        if isinstance(time_info, (int, float)):
            val,_,_ = self._get_random([time_info])
            if val.size == 1:
                return val[0]
            else:
                return None
        times = self._time2arr_get(time_info)
        start, stop, step = self._to_slice(times)
        if not self._is_sorted(times):
            return self._get_random(times)
        if not self.step or not step:
            return self._get_variable(times)

        before = (times < self.start)
        after = (times > self.stop)
        flag = [np.any(after), np.any(before)]

        ratio = step / self.step
        start_idx = (start - self.start) / self.step
        ratio_isint = self._is_int(ratio)
        start_isint = self._is_int(start_idx)

        if (ratio_isint and start_isint):
            flag += [False]
            ratio_int = int(np.round(ratio))
            start_int = int(np.round(start_idx))
            start_int = start_int if start_int >= 0 else 0
            stop_int = start_int + ratio_int * (len(times))
            return (self.vals[start_int:stop_int:ratio_int],
                    self.times[start_int:stop_int:ratio_int],
                    times[np.logical_or(before, after)], flag)

        if self.interpolation:
            raise NotImplementedError

        flag += [np.sum(before) + np.sum(after) != len(times)]
        return (np.array([], dtype=object), np.array([], dtype=float),
                times, flag)

    def missing(self, time_info):
        if self.N == 0:
            return time_info
        times = self._time2arr_get(time_info)
        if not self._is_sorted(times):
            times = np.sort(times)
        outside = np.logical_or(times < self.start,
                                times > self.stop)
        times_in = times[np.logical_not(outside)]
        start, stop, step = self._to_slice(times_in)
        if self.interpolation:
            return times[outside]
        if step and self.step:
            ratio_int = self._is_int(step / self.step)
            start_int = self._is_int((start - self.start) /
                                     self.step)
            stop_int = self._is_int((self.stop - stop) /
                                    self.step)
            if (ratio_int and start_int and stop_int):
                return times[outside]
        no_match = self._no_match(times_in)
        return np.sort(np.concatenate([times[outside], no_match]))

    def _sort(self):
        # ensure times are sorted
        if self._is_sorted(self.times):
            return # already sorted
        sorted_idx = np.argsort(self.times)
        self.times = self.times[sorted_idx]
        self.vals = self.vals[sorted_idx]

    def _remove_donblons(self):
        # ensure no double entree at one time.
        # call only once sorted
        if len(self.times) < 2 or not any(np.diff(self.times)==0):
            return
        _, unique_idx = np.unique(self.times, return_index=True)
        self.times = self.times[unique_idx]
        self.vals = self.vals[unique_idx]

    def _no_match(self, times):
        missing = [t for t in times if not np.any(np.isclose(t, self.times))]
        return np.array(missing)

    def _get_random(self, times):
        vals = []
        ins = []
        misses = []
        flags = [False, False, False]
        for t in times:
            idx = np.argmin(np.abs(self.times - t))
            delta = self.times[idx] - t
            if np.abs(delta) <= 1e-8:
                ins.append(self.times[idx])
                vals.append(self.vals[idx])
            elif self.interpolation and t > self.start and t < self.stop:
                raise NotImplementedError
            else:
                if t < self.start:
                    flags[0] = True
                elif t > self.stop:
                    flags[1] = True
                else:
                    flags[2] = True
                misses.append(t)
        return (np.array(vals, dtype=object), np.array(ins),
                np.array(misses), flags)

    def _get_variable(self, times):
        before = times < self.start
        after = times > self.stop
        flag = [np.any(after), np.any(before)]
        outside = np.logical_or(before, after)
        times_before = times[before]
        times_after = times[after]
        times_in = times[np.logical_not(outside)]
        vals, times_ok, miss, rflag = self._get_random(times_in)
        miss = np.concatenate([times[before], miss, times[after]])
        flag += [rflag[2]]
        return vals, times_ok, miss, flag

    def _time2arr(self, time_info):
        if isinstance(time_info, (int, float)):
            times = np.array([time_info])
        elif isinstance(time_info, (list, np.ndarray)):
            times = np.array(time_info)
        return times

    def _time2arr_get(self, time_info):
        if not isinstance(time_info, slice):
            times = self._time2arr(time_info)
        else:
            start = time_info.start if time_info.start is not None else self.start
            stop = time_info.stop if time_info.stop is not None else self.stop
            if time_info.step is not None:
                stop = int((stop-start)/time_info.step)*time_info.step+start
                times = np.linspace(start, stop, 1 + (stop - start) / (time_info.step))
            else:
                times = self.times[np.logical_and(self.times >= start, self.times <= stop)]
        return times

    def _time2arr_set(self, time_info, N):
        if not isinstance(time_info, slice):
            times = self._time2arr(time_info)
        else:
            start = time_info.start
            stop = time_info.stop if time_info.stop is not None else start + time_info.step * (N-1)
            times = np.linspace(start, stop, N)
        return times

    @staticmethod
    def _to_slice(times):
        # sorted array to start, stop, N, dt
        # dt == 0 if step size not constant
        start = times[0]
        stop = times[-1]
        N = len(times)
        if N == 1:
            step = 0
        elif all(np.diff(times) == (stop-start)/(N-1)):
            step = (stop-start)/(N-1)
        else:
            step = 0
        return start, stop, step

    @staticmethod
    def _is_sorted(arr):
        return len(arr) < 2 or all(np.diff(arr)>=0)

    @staticmethod
    def _is_int(number):
        return np.abs(np.round(number) - number) < 1e-12


class _SolverCacheOneLevel:
    def __init__(self, parent, key, maker_func, t0, level):
        self.this_level = _SolverCacheOneEvo()
        self.keys = []
        self.caches = []
        self.maker_func = maker_func[0]
        self.child_maker_func = maker_func[1:]
        self.parent = parent
        self.key = key
        self.level = level
        self.t0 = t0
        if self.level == 1:
            self.this_level[t0] = self.key

    def __setitem__(self, args, vals):
        if isinstance(args, tuple) and len(args) >=2:
            key = args[0]
            other_keys = args[1:]
            if key in self.keys:
                idx = self.keys.index(key)
                self.caches[idx][other_keys] = vals
            else:
                self.keys.append(key)
                self.caches.append(_SolverCacheOneLevel(self, key,
                                                        self.child_maker_func,
                                                        self.t0, self.level+1))
                self.caches[-1][other_keys] = vals
        else:
            args = args[0] if isinstance(args, tuple) else args
            self.this_level[args] = vals

    def __getitem__(self, args):
        if isinstance(args, tuple) and len(args) >=2:
            key = args[0]
            other_keys = args[1:]
            if key in self.keys:
                return self.caches[self.keys.index(key)][other_keys]
            else:
                self.keys.append(key)
                self.caches.append(_SolverCacheOneLevel(self, key,
                                                        self.child_maker_func))
                return self.caches[-1][other_keys]
        else:
            args = args[0] if isinstance(args, tuple) else args
            missing = self.this_level.missing(args)
            if missing.size != 0:
                source, times, _, _ = self.parent[missing]
                self.this_level[times] = self.maker_func(source,
                                                         times, self.key)
            return self.this_level[args]

    def first_after(self, t, state=None):
        if state is None:
            times = self.this_level.times
            t_ = times[((times - t) > 0)[0]]
            val = self.this_level[t_]
        else:
            times = self.caches[self.keys.index(state)]
            t_ = times[((times - t) > 0)[0]]
            val = self.caches[self.keys.index(state)][t_]
        return val, t_

    def last_before(self, t, state=None):
        if state is None:
            times = self.this_level.times
            t_ = times[((t - times) > 0)[-1]]
            val = self.this_level[t_]
        else:
            times = self.caches[self.keys.index(state)]
            t_ = times[((t - times) > 0)[-1]]
            val = self.caches[self.keys.index(state)][t_]
        return val, t_


class _SolverCache:
    def __init__(self, solver, args):
        self.num_args = 0
        self.args_hash = {}
        self.cached_data = []
        self.solver = solver
        self.args = args
        self.argsk = [key for key in args.keys if not is_dynargs_pattern(key)]

    def _new_cache(self, args):
        funcs = [None, _prop2state, _expect]
        cache = _SolverCacheOneLevel(dummy_parent, args,
                                     funcs, self.t_start, 0)
        self.solver.LH.arguments(args)
        dims = self.solver.dims
        cache[self.solver.t_start] = qeye(dims[0])
        return cache

    def _hashable_args(self, args):
        fullargs = self.args.copy()
        fullargs.update(args)
        args_values = tuple(fullargs[key] for key in self.argsk)
        return args_values

    def __setitem__(self, key, val):
        args = self._hashable_args(key[0])
        other_key = key[1:]
        if args not in self.args_hash:
            self.cached_data.append(self._new_cache(args))
            self.args_hash[args] = self.num_args
            self.num_args += 1
        self.cached_data[self.args_hash[args]][other_key] = val

    def __getitem__(self, key):
        args = self._hashable_args(key[0])
        other_key = key[1:]
        if args not in self.args_hash:
            self.cached_data.append(self.new_cache(args))
            self.args_hash[args] = self.num_args
            self.num_args += 1
        return self.cached_data[self.args_hash[args]][other_key]

    def add_prop(self, props, args, times):
        self[(args, times)] = props

    def get_prop(self, args, times):
        return self[(args, times)]

    def need_compute_prop(self, args, times):
        return self[(args, times)][2:4]

    def add_state(self, states, args, state, times):
        self[(args, psi, times)] = states

    def get_state(self, args, state, times):
        return self[(args, psi, times)]

    def need_compute_state(self, args, state, times):
        return self[(args, psi, times)][2:4]

    def add_expect(self, values, args, state, e_ops, times):
        if isinstance(e_ops, list):
            for e_op, val in zip(e_ops, values):
                self[(args, psi, e_op, times)] = val
        else:
            self[(args, psi, e_ops, times)] = values

    def get_expect(self, args, state, e_ops, times):
        if isinstance(e_ops, list):
            expects = [self[(args, psi, e_op, times)]
                       for e_op, val in zip(e_ops, values)]
        else:
            expects = self[(args, psi, e_ops, times)]
        return expects

    def need_compute_expect(self, args, state, e_ops, times):
        if isinstance(e_ops, list):
            expects = [self[(args, psi, e_op, times)][2:4]
                       for e_op, val in zip(e_ops, values)]
        else:
            expects = self[(args, psi, e_ops, times)][2:4]
        return expects

    def first_after(self, t, args, state=None):
        if state is None:
            val, ts, _, after = self.get_prop([args, t])
        else:
            val, ts, _, after = self.get_state([args, state, t])
        if ts.size == 1:
            return val, ts
        args = self._hashable_args(args)
        return self.cached_data[self.args_hash[args]].first_after(t, state)

    def last_before(self, t, args, state=None):
        if state is None:
            val, ts, before, _ = self.get_prop([args, t])
        else:
            val, ts, before, _ = self.get_state([args, state, t])
        if ts.size == 1:
            return val, ts
        args = self._hashable_args(args)
        return self.cached_data[self.args_hash[args]].last_before(t, state)


class dummy_parent:
    def __getitem__(self, args):
        return (np.array([], dtype=object), np.array([]),
                args, [True, False, True])


def _prop2state(self, sources, times, key):
    return np.array([prop * key for prop in sources], dtype=object)


def _expect(self, sources, times, key):
    return np.array([key.expect(t, psi)
                     for psi, t in zip(sources, times)], dtype=object)
