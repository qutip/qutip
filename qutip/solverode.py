


import numpy as np
from scipy.integrate import solve_ivp, ode
from qutip.solver import

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
    OdeScipyAdam:

    OdeScipyDopri:

    OdeQutipDopri:

    OdeQutipAdam:

    OdeDiagonalized:

    OdeSparse:

    OdeAdaptativeHilbertSpace:

    OdeFloquet:

    OdeMonteCarlo:

    """
    def __init__(self, L, opt, e_ops, ode_args,
                 normalize_func, progress_bar):
        self.opt = opt
        self.L = L
        self.e_ops = e_ops
        self.ode_args = ode_args
        self.normalize_func = normalize_func
        self.progress_bar = progress_bar

    def run(self, state0, tlist, args={}):
        raise NotImplementedError


class OdeScipyAdam(OdeSolver):
    # -------------------------------------------------------------------------
    # Solve an ODE for func.
    # Calculate the required expectation values or invoke callback
    # function at each time step.
    def run(self, state0, tlist, args={}):
        """
        Internal function for solving ODEs.
        """
        opt = self.opt
        normalize_func = self.normalize_func
        self.L.arguments(args, state=state0,
                         e_ops=self.e_ops.raw_e_ops)
        func = self.L._get_mul(state0)
        n_tsteps = len(tlist)
        if opt.store_states:
            states = np.zeros((n_tsteps, state0.shape[0]*state0.shape[1]),
                              dtype=complex)
        else:
            states = np.zeros((1, state0.shape[0]*state0.shape[1]),
                              dtype=complex)

        r = ode(func)
        r.set_integrator('zvode', method=opt.method, order=opt.order,
                         atol=opt.atol, rtol=opt.rtol, nsteps=opt.nsteps,
                         first_step=opt.first_step, min_step=opt.min_step,
                         max_step=opt.max_step)
        if self.ode_args:
            r.set_f_params(*self.ode_args)
        initial_vector = state0.full().ravel('F')
        r.set_initial_value(initial_vector, tlist[0])

        e_ops_store = bool(self.e_ops)
        self.e_ops.init(tlist)
        self.progress_bar.start(n_tsteps-1)
        for t_idx, t in enumerate(tlist):
            if not r.successful():
                raise Exception("ODE integration error: Try to increase "
                                "the allowed number of substeps by increasing "
                                "the nsteps parameter in the Options class.")
            # get the current state / oper data if needed
            if opt.store_states or opt.normalize_output or e_ops_store:
                cdata = r.y
            if opt.normalize_output:
                norm = normalize_func(cdata)
                if norm > 1e-12:
                    r.set_initial_value(cdata, r.t)
                else:
                    r._y = cdata
            if opt.store_states:
                states[t_idx, :] = cdata
            self.e_ops.step(t_idx, cdata)
            if t_idx < n_tsteps - 1:
                r.integrate(tlist[t_idx+1])
                self.progress_bar.update(t_idx)
        self.progress_bar.finished()
        if not opt.store_states:
            states[0, :] = r.y
        return states, self.e_ops.finish()


class OdeScipyIVP(OdeSolver):
    # -------------------------------------------------------------------------
    # Solve an ODE for func.
    # Calculate the required expectation values or invoke callback
    # function at each time step.
    def run(self, state0, tlist, args={}):
        """
        Internal function for solving ODEs.
        """
        opt = self.opt
        self.func.arguments(args, state=state0,
                            e_ops=self.e_ops.raw_e_ops)
        func = self.L._get_mul(state0)
        if opt.store_states:
            states = np.zeros((len(tlist), state0.shape[0]*state0.shape[1]),
                              dtype=complex)
        else:
            states = np.zeros((1, state0.shape[0]*state0.shape[1]),
                              dtype=complex)

        initial_vector = state0.full().ravel('F')
        ivp_opt = {method=opt.method, atol=opt.atol, rtol=opt.rtol,
                   min_step=opt.min_step, max_step=opt.max_step,
                   first_step=opt.first_step}

        ode_res = solve_ivp(func, [tlist[0], tlist[-1]], initial_vector,
                            t_eval=tlist, args=self.ode_args, event=event,
                            **ivp_opt)

        self.e_ops.init(tlist)
        for t_idx, cdata in enumerate(ode_res.y):
            if opt.store_states:
                states[t_idx, :] = cdata
            self.e_ops.step(t_idx, cdata)
        if not opt.store_states:
            states[0, :] = cdata
        return states, self.e_ops.finish()


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
                                                        self.child_maker_func
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
    def __init__(self, solver):
        self.num_args = 0
        self.args_hash = {}
        self.cached_data = []
        self.solver = solver

    def _new_cache(self, args):
        funcs = [None, _prop2state, _expect]
        cache = _SolverCacheOneLevel(dummy_parent, args,
                                     funcs, self.t_start, 0)
        self.solver.LH.arguments(args)
        dims = self.solver.dims
        cache[self.solver.t_start] = qeye(dims[0])
        return cache

    def _hashable_args(self, args):
        args, dyn_args = args[0], args[1]
        dyn_list = tuple(sorted([dyn[0] for dyn in dyn_args]))
        # collapse? e_ops change?
        keys = [key for key in args.keys() if key not in dyn_list]
        keys = tuple(sorted(keys))
        args_tuple = tuple((args[key] for key in keys))
        return (args_tuple, dyn_list)

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
            for e_op, val in zip(e_ops, values)
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
