


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
        print(times)
        start, stop, step = self._to_slice(times)
        if not self._is_sorted(times):
            return self._get_random(times)
        if not self.step or not step:
            return self._get_variable(times)

        before = (times < self.start)
        after = (times > self.stop)

        ratio = step / self.step
        start_idx = (start - self.start) / self.step
        ratio_isint = self._is_int(ratio)
        start_isint = self._is_int(start_idx)

        if (ratio_isint and start_isint):
            ratio_int = int(np.round(ratio))
            start_int = int(np.round(start_idx))
            start_int = start_int if start_int >= 0 else 0
            stop_int = start_int + ratio_int * (len(times))
            return (self.vals[start_int:stop_int:ratio_int],
                    self.times[start_int:stop_int:ratio_int],
                    times[np.logical_or(before, after)])

        if self.interpolation:
            raise NotImplementedError

        return np.array([], dtype=object), np.array([], dtype=float), times

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
        for t in times:
            idx = np.argmin(np.abs(self.times - t))
            delta = self.times[idx] - t
            if np.abs(delta) <= 1e-8:
                ins.append(self.times[idx])
                vals.append(self.vals[idx])
            elif self.interpolation and t > self.start and t < self.stop:
                raise NotImplementedError
            else:
                misses.append(t)
        return np.array(vals, dtype=object), np.array(ins), np.array(misses)

    def _get_variable(self, times):
        before = times < self.start
        after = times > self.stop
        outside = np.logical_or(before, after)
        times_before = times[before]
        times_after = times[after]
        times_in = times[np.logical_not(outside)]
        vals, times_ok, miss = self._get_random(times_in)
        miss = np.concatenate([times[before], miss, times[after]])
        return vals, times_ok, miss

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
    def __init__(self, parent, key, maker_func):
        self.this_level = _SolverCacheOneEvo()
        self.keys = []
        self.caches = []
        self.maker_func = maker_func[0]
        self.child_maker_func = maker_func[1:]
        self.parent = parent
        self.key = key

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
                                                        self.child_maker_func))
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
                source, times, _ = self.parent[missing]
                self.this_level[times] = self.maker_func(source, self.key)
            return self.this_level[args]

class _SolverCache:
    def __init__(self, t_start=0, dt=1):
        self.num_args = 0
        self.args_hash = {}
        self.cached_data = []
        self.t_start = t_start
        self.dt = dt

    def _hashable_args(self, args, dyn_args):
        dyn_list = tuple(sorted([dyn[0] for dyn in dyn_args]))
        # collapse? e_ops change?
        keys = [key for key in args.keys() if key not in dyn_list]
        keys = tuple(sorted(keys))
        args_tuple = tuple((args[key] for key in keys))
        return (args_tuple, dyn_list)

    def __setitem__(self, key, val):
        args = key[0]
        other_key = key[1:]
        if args not in self.args_hash:
            self.cached_data.append(self.new_cache(args))
            self.args_hash[args] = self.num_args
            self.num_args += 1
        self.cached_data[self.args_hash[key]][other_key] = val

    def __getitem__(self, key):
        args = key[0]
        other_key = key[1:]
        if args not in self.args_hash:
            self.cached_data.append(self.new_cache(args))
            self.args_hash[args] = self.num_args
            self.num_args += 1
        return self.cached_data[self.args_hash[key]][other_key]

    def add_prop(self, args, props, t_start=None, dt=None):
        t_start = t_start if t_start is not None else self.t_start
        dt = dt if dt is not None else self.dt
        t_end = len(props) * dt + t_start
        argsCache = self[args]
        if argsCache.get_prop_evolution() is None:
            argsCache.prop = _SolverCacheOneEvo(t_start, t_end, dt, props)
        else:
            propCache = argsCache.get_prop_evolution()
            propCache.add(t_start, t_end, dt, props)

    def need_compute_prop(self, args, t_end, t_start=None, dt=None):
        t_start = t_start if t_start is not None else self.t_start
        dt = dt if dt is not None else self.dt
        argsCache = self[args]
        if argsCache.get_prop_evolution() is None:
            return t_start # Need to be computed from the beginning
        else:
            propCache = argsCache.get_prop_evolution()
            return propCache.restart(t_start, t_end, dt)

    def get_prop(self, args, t_end, t_start=None, dt=None):
        if t_start is not None else self.t_start:
            t_start = t_start
            t_end = t_end + 1
        dt = dt if dt is not None else self.dt
        argsCache = self[args]
        if argsCache.get_prop_evolution() is None:
            return None
        else:
            propCache = argsCache.get_prop_evolution()
            return propCache(t_start, t_end, dt)

    def add_state(self, args, psi, states, t_start=None, dt=None):
        t_start = t_start if t_start is not None else self.t_start
        dt = dt if dt is not None else self.dt
        t_end = len(props) * dt + t_start
        argsCache = self[args]
        if argsCache.get_state_evolution(psi) is None:
            argsCache.psi0s.append(psi)
            argsCache.psis.append(_SolverCacheOneEvo(t_start, t_end,
                                                     dt, states))
        else:
            stateCache = argsCache.get_state_evolution(psi)
            stateCache.add(t_start, t_end, dt, states)

    def _need_compute_state(self, args, psi, t_end, t_start=None, dt=None):
        t_start = t_start if t_start is not None else self.t_start
        dt = dt if dt is not None else self.dt
        argsCache = self[args]
        if argsCache.get_state_evolution(psi) is None:
            as_state = t_start # Need to be computed from the beginning
        else:
            stateCache = argsCache.get_state_evolution(psi)
            as_state = stateCache.restart(t_start, t_end, dt)
        return as_state

    def need_compute_state(self, args, psi, t_end, t_start=None, dt=None):
        as_state = self._need_compute_prop(args, psi, t_end, t_start, dt)
        as_prop = t_start
        if as_state < t_end:
            as_prop = self.need_compute_prop(args, t_end, t_start, dt)
        return max(as_state, as_prop)

    def get_state(self, args, psi, t_end, t_start=None, dt=None):
        if t_start is not None else self.t_start:
            t_start = t_start
            t_end = t_end + 1
        dt = dt if dt is not None else self.dt
        argsCache = self[args]
        as_state = self._need_compute_prop(args, psi, t_end, t_start, dt)

        if argsCache.get_state_evolution(psi) is None:
            states = []
        else:
            stateCache = argsCache.get_state_evolution(psi)
            states = stateCache(t_start, as_state, dt)

        if as_state < t_end:
            propCache = argsCache.get_state_evolution(psi)
            props = propCache(as_state+dt, t_end, dt)
            new_states += [prop*psi for prop in props]
            self.add_state(args, psi, new_states, t_end, as_state+dt, dt)
            states += new_states

        return states

    def add_expect(self, args, psi, e_ops, values, t_start=None, dt=None):
        t_start = t_start if t_start is not None else self.t_start
        dt = dt if dt is not None else self.dt
        t_end = len(props) * dt + t_start
        argsCache = self[args]
        if argsCache.get_expect_evolution(psi, e_ops) is None:
            self.e_ops_val = [[]]
            self.e_psis = []
            self.e_ops = []
            if psi not in self.e_psis:
                self.e_psis.append(psi)
                self.e_ops_val.append([None]*len(self.e_ops))
            if e_ops not in self.e_ops:
                self.e_ops.append(e_ops)
                for one_psi in self.e_ops_val:
                    one_psi.append(None)

            i = self.e_psis.index(psi)
            j = self.e_ops.index(psi)
            self.e_ops_val[i][j]= _SolverCacheOneEvo(t_start, t_end,
                                                     dt, values)
        else:
            expectCache = argsCache.get_expect_evolution(psi, e_ops)
            expectCache.add(t_start, t_end, dt, states)

    def _need_compute_expect(self, args, psi, e_ops, t_end, t_start=None, dt=None):
        t_start = t_start if t_start is not None else self.t_start
        dt = dt if dt is not None else self.dt
        argsCache = self[args]
        if argsCache.get_expect_evolution(psi, e_ops) is None:
            as_expect = t_start # Need to be computed from the beginning
        else:
            expectCache = argsCache.get_expect_evolution(psi, e_ops)
            as_expect = expectCache.restart(t_start, t_end, dt)
        return as_expect

    def need_compute_expect(self, args, psi, e_ops, t_end, t_start=None, dt=None):
        as_expect = self._need_compute_expect(args, psi, e_ops, t_end, t_start, dt)
        as_state = t_start
        if as_expect < t_end:
            as_state = self.need_compute_state(args, t_end, t_start, dt)
        return max(as_state, as_expect)

    def get_expect(self, args, psi, e_ops, t_end, t_start=None, dt=None):
        if t_start is not None else self.t_start:
            t_start = t_start
            t_end = t_end + 1
        dt = dt if dt is not None else self.dt
        argsCache = self[args]
        as_expect = self._need_compute_expect(args, psi, t_end, t_start, dt)

        if argsCache.get_expect_evolution(psi, e_ops) is None:
            expects = []
        else:
            expectCache = argsCache.get_expect_evolution(psi, e_ops)
            expects = expectCache(t_start, as_state, dt)

        if as_expect < t_end:
            stateCache = argsCache.get_state_evolution(psi)
            states = stateCache(as_state+dt, t_end, dt)
            new_expects += [e_ops.expect(state) for state in states]
            self.add_expect(args, psi, e_ops, new_expects, t_end, as_state+dt, dt)
            expects += new_expects

        return expects























class _SolverCacheOneEvoOld:
    def __init__(self, t_start, dt, vals):
        self.t_start = t_start
        self.dt = dt
        self.vals = vals
        self.dense = False
        self.num_val = len(self.vals)

    def _index_from_time(t):
        return np.round((t-self.t_start)/self.dt, 5)

    def __contains__(self, t):
        index = _index_from_time(t)
        if index < 0 or index >= self.num_val:
            return False
        if index != np.round(index):
            return False
        return True

    def add(self, t_start, vals):
        index = _index_from_time(t_start)
        if index != np.round(index):
            raise ValueError("Can't add new data")
        if index == 0:
            self.vals = vals
            self.num_val = len(self.vals)
        elif index = self.num_val:
            self.vals.append(vals)
            self.num_val = len(self.vals)
        elif index > 0 and index < self.num_val:
            self.vals = self.vals[:index] + vals
            self.num_val = len(self.vals)
        else:
            raise ValueError("Can't add new data")

    def has(self, t_start, t_end=None, dt=0):
        if t_end is None:
            return t_start in self
        dt = np.round(dt/self.dt, 5) if dt else 1
        return t_start in self and t_end in self and dt == np.round(dt)

    def __call__(self, t_start, t_end=None, dt=0):
        # Same structure as deference: [start: end: step]
        # However 'start' is mandatory
        index_start = _index_from_time(t_start)
        if t_end is None and not dt:
            return self.vals[index]
        index_end = self.num_val if t_end is None else _index_from_time(t_end)
        step = np.round(dt/self.dt, 5) if dt else 1
        return self.vals[np.rint(index_start):np.rint(index_end):np.rint(step)]

    def restart(self, t_start, t_end, dt):
        if t_start not in self:
            return (t_start, None)
        index_start = _index_from_time(t_start)
        index_end = _index_from_time(t_end)
        step = np.round(dt / self.dt, 5)
        if step != np.round(step):
            return (t_start, self(t_start))
        if index_end < self.num_val:
            return (t_end, None)
        last_position = ((self.num_val - 1 - t_start)//dt) * dt + t_sta
        return (last_position, self.vals[last_position])

    def __bool__(self):
        return bool(self.vals)
