

import numpy as np
import matplotlib.pyplot as plt
from qutip import mcsolve, sesolve, mesolve, steadystate
from qutip import Options, BaseProgressBar


class Qsystem:
    def __init__(self, H, c_ops=[], args={}, e_ops=[]):
        # system
        self.H = H
        self.c_ops = c_ops
        self.args = args
        self.e_ops = e_ops
        self.L = None

        # state
        self.t = 0.
        self.t0 = 0.
        self.state_0 = None
        self.state_t = None
        self.states_runs_t = []
        self._tlist = []

        # options
        self.progress_bar = BaseProgressBar()
        self._options = Options()
        self.ntraj = [1]

        # self.optimization
        self.period = 0
        self.diagonalized = False
        self.sparsity = 1

    def reset(self, t=0, state=None, tlist=[]):
        if tlist:
            self._tlist = _tlist
            self.t0 = _tlist[0]
            self.t = _tlist[0]
        else:
            self.t0 = t
            self.t = self.t0

        self._check_consistance(state)
        self.state_0 = state
        self.state_t = None
        self.states_runs_t = []

    def evolve(self, tlist=[], state0=None, t=None):
        if tlist:
            self.tlist = tlist
        if not self._tlist:
            smart_tlist = True
        else:
            smart_tlist = False

        if state0 is None:
            state0 = self.state_0

        if self.ready:
            self.prepare()

        if self.evolve_type == "me":
            # use mesolve
            res = mesolve(self.H, state0, tlist, self.c_ops,
                          e_ops=self._e_ops,
                          args=self.args, options=self.options,
                          progress_bar=self.progress_bar)

        elif self.evolve_type == "se":
            # use sesolve
            res = sesolve(self.H, state0, tlist, e_ops=self._e_ops,
                          args=self.args, options=self.options,
                          progress_bar=self.progress_bar)

        elif self.evolve_type == "mc":
            # use sesolve
            res = mcsolve(self.H, state0, tlist, self.c_ops,
                          e_ops=self._e_ops, args=self.args, ntraj=self.ntraj,
                          options=self.options,
                          progress_bar=self.progress_bar)

        self.state_t = res.final_state
        self.t = tlist[-1]
        self.evolution_results = res

    def get_trajectories(self, ntraj=100, tlist=[], state0=None, t=None, dt=None):
        if not np.arraylike(tlist):
            if t is not None and dt is not None:
                tlist = np.linspace(selt.t, t, (t-self.t)//dt+1)
                if state0 is None and self.state_t is not None:
                    state0 = self.state_t
            else:
                raise TypeError("tlist expected")
        if state0 is None:
            state0 = self.state_0

        res = mcsolve(H, state0, tlist, c_ops=self.c_ops, e_ops=self._e_ops,
                      ntraj=ntraj, args=self.args, options=self.options,
                      progress_bar=self.progress_bar)
        self.state_t = res.final_state
        self.t = tlist[-1]
        self.path_results = res

    def steadystate(method='direct', solver=None, **kwargs):
        ss = steadystate(self.H, self.c_ops, method=method, solver=solver,
                         **kwargs)
        self.ss = ss

    def plot_expect(self):
        if self.evolution_results:
            plot_expectation_value(self.evolution_results)
        if self.path_results:
            plot_expectation_value(self.path_results)
        if self.ss:
            expect_fs = []
            for e_op in self.e_ops:
                expect_fs.append(expect(e_op, self.ss))
                plt.axhline(y=expect_fs[-1], lw=1.5)

    def optimization(self, period=0, diagonalize=False, sparse_state=False):
        raise NotImplementedError

    def expect(self):
        self.evolution_results.expect

    def states(self):
        self.path_results.expect

    def runs_expect(self):
        self.path_results.expect

    def runs_expect_averages(self):
        self.path_results.expect

    def runs_states(self):
        self.path_results.expect

    def final_state(self):
        pass

    def runs_final_states(self):
        pass


class SESolver(Solver):
    """Schrodinger Equation Solver

    """
    def __init__(self, H, args=None,
                 tlist=None, state0=None, e_ops=None,
                 options=None, cache_level="state", progress_bar=True):
        # TODO: split options?
        self.H = H
        self._args0 = args if args is not None else {}
        self._tlist0 = tlist
        self._state0 = state0
        self._e_ops0 = e_ops
        self._prepare_H()

        self.options = options
        self.progress_bar = progress_bar
        self.map_func = map_func

        self.prepare_cache(self)
        self._cache(self)
        self.optimization = {"diagonal":False, "period":0}
        self.solver = None

    @property
    def options(self):
        return self._options

    @options.setter
    def options(self, options):
        if isinstance(options, Options):
            self._options = options
            self._prepare_solver()
        else:
            raise TypeError("Invalid options")

    @property
    def map_func(self):
        return self._map_func

    @map_func.setter
    def map_func(self, map_func):
        if map_func in [parallel_map, serial_map]:
            self._map_func = map_func
        else:
            raise TypeError

    @property
    def progress_bar(self):
        return self._progress_bar

    @progress_bar.setter
    def progress_bar(self, progress_bar):
        if isinstance(progress_bar, BaseProgressBar):
            self._progress_bar = progress_bar
        elif progress_bar:
            self._progress_bar = TextProgressBar
        else:
            self._progress_bar = BaseProgressBar

    def _prepare_H(self):
        H_td = -1.0j * qobjevo_maker(H, args=self._args0, tlist=self._tlist0,
                                     state=self._psi0, e_ops=self._e_ops0)
        nthread = self.options.openmp_threads if self.options.use_openmp else 0
        H_td.compile(omp=nthread, dense=options.use_dense_matrix)
        self.LH = H_td
        self.dims = H_td.dims
        self.shape = H_td.shape
        self.has_dyn_args = len(H_td.dynamics_args)

    def _prepare_solver(self):
        if self.optimization["diagonal"]:
            raise NotImplementedError
        elif self.optimization["period"]:
            raise NotImplementedError
        elif self.option.method in ['adams','bdf']:
            solver = OdeScipyAdam
        else:
            solver = OdeScipyIVP
        self.LH_Solver = solver.prepare(self.LH, self.options)
        self.solver = solver

    def _get_solver(self, args, progress_bar=None):
        solver = self.solver(self.LH_Solver, self.options)
        solver.progress_bar = progress_bar
        return solver

    def prepare_cache(self, cache_level):
        if isinstance(cache_level, str):
            cache_level = cache_level.casefold()

        if cache_level in [4, "full"]:
            # Cache safe up to propagators
            self._cache_level = 4
        elif cache_level in [3, True, "state", "states", "psi", "rho"]:
            # Cache safe up to state
            self._cache_level = 3
        elif cache_level in [2, "state only"]:
            # Cache only states
            self._cache_level = 2
        elif cache_level in [1, "expect"]:
            # Cache safe up to propagators
            self._cache_level = 1
        elif cache_level in [0, False, None, "none"]:
            # Cache safe up to propagators
            self._cache_level = 0
        else:
            raise ValueError("cache_level must be one of 'full', 'state', "
                             "'state only', 'expect', 'none'")
        if self.has_dyn_args and self._cache_level == 4:
            # With feedback, the propagator is not valid
            self._cache_level = 3

    def optimization(self, **optimizations):
        raise NotImplementedError

    def propagator(self, t, args=None, dtype="Qobj", starts=None):
        # TODO: With feedback, the propagator is not valid
        # Raise error is self.has_dyn_args?
        args_sets, num_args = self._fix_input_args(args)
        tlist, num_t = self._fix_input_time(t)

        if starts is None:
            res_time = self.prop_core(tlist, args_sets)
            results = self._sort_by_tlist(res_time, tlist, 1)
        else:
            start_tlist, num_t2 = self._fix_input_time(t)
            res, time = self._prop_core(tlist + start_tlist, args_sets)
            results = [self._dprop(by_args, time, tlist, start_tlist)
                       for by_args in res]

        results = self._apply_type(results, dtype, None, 1)
        if not num_t and starts is None or not num_t2:
            results = [res_set[0] for res_set in results]
        if not num_args:
            results = results[0]
        return results

    def state(self, t, state0=None, args=None,  dtype="Qobj"):
        """
        Give the state at time t given an initial state0 at t=0.
        If the Hamiltonian is defined with arguments, the default can be
        overwritten.

        t, state0, args can be list:
            results[a, s, t] = state(t, state0[s], args[a])

        """
        tlist, num_t = self._fix_input_time(t)
        state, num_state = self._fix_input_state(state0)
        args, num_args = self._fix_input_args(args)

        run_set = []
        run_idx = []
        res = []
        if num_state >= self.shape[0]/2 and self._cache_level >= 4:
            # Evolve propagators to compute state from them.
            # Calling 'propagator' will automatically cache them.
            self.propagator(tlist, args, dtype="")

        times = np.unique(np.sort(tlist))
        for i, arg in enumerate(args_sets):
            res.append([None] * len(num_state))
            for j, state in enumerate(state):
                prop, times, past, futur = \
                    self._cache.get_state(arg, state, times)
                if past.size > 0:
                    state_f, t = self._cache.first_after(np.max(past), state)
                    run_set.append([(arg, state), past, state_f, t])
                    run_idx.append((i,j))
                if futur.size > 0:
                    state_l, t = self._cache.last_before(np.min(futur), state)
                    run_set.append([arg, state, futur, state_l, t])
                    run_idx.append((i,j))
                if times > 0:
                    res[i][j] = [[prop, times]]
                else:
                    res[i][j] = []

        runs = self._run_sets(run_set)
        for (i, j), run in zip(run_idx, runs):
            res[i][j].append(run)

        results = self._clean_res(arg_set, 2)
        results = self._sort_by_tlist(results, tlist, 2)
        results = self._apply_type(results, dtype, None, 2)

        if not num_t:
            results = [[by_state[0] for by_state in by_args]
                       for by_args in results]
        if not num_state:
            results = [by_args[0] for by_args in results]
        if not num_args:
            results = results[0]
        return results

    def expect(self, t, state0=None, args=None, e_ops=None):
        if self.cache_level <= 1:
            return self._raw_expect(t, state0, args, e_ops)

        # Compute and save states.
        self.state(t, state0, args)

        tlist, num_t = self._fix_input_time(t)
        e_ops, num_e_ops = self._fix_input_e_ops(e_ops)
        states, num_state = self._fix_input_state(state)
        args_sets, num_args = self._fix_input_args(args)

        res = []
        for arg in args_sets:
            one_args = []
            for state in states:
                one_state0 = []
                for e_op in e_ops:
                    exp, times, past, futur = \
                        self._cache.get_state(arg, state, e_op, tlist)
                    if past.size > 0 or futur.size > 0:
                        raise Exception("Should be empty...")
                    one_state0.append(exp if num_t > 1 else exp[0])
                one_args.append(one_state0 if num_e_ops > 1 else one_state0[0])
            res.append(one_args if num_state > 1 else one_args[0])
        expects = np.array(res)
        if num_args == 1:
            expects = expects[0]
        return expects

    def plot_expect(self, tlist, psi0=None, args=None, e_op=None, fig=None):
        pass


    def _fix_input_args(self, args):
        if args is None:
            args = self._args0
        if not isinstance(args, list):
            args = [args]
            num_args = 0
        else:
            num_args = len(args)
        return args, len(args)

    def _fix_input_time(self, times):
        if isinstance(args, np.ndarray):
            times = list(times)
        if not isinstance(times, list):
            times = [times]
            num = 0
        else:
            num = len(times)
        return times, num

    def _fix_input_state(self, state):
        if state is None:
            state = self._state0
        if not isinstance(state, list):
            state = [state]
            num = 0
        else:
            num = len(state)
        return state, num

    def _fix_input_e_ops(self, e_ops):
        if e_ops is None:
            e_ops = self._e_ops0
        if not isinstance(e_ops, list):
            e_ops = [e_ops]
            num = 0
        else:
            num = len(e_ops)
        return e_ops, num

    def _clean(self, results, depth):
        """ If new results were partialy cached,
        merge the cached and new values.
        """
        if depth:
            # Get to the right depth in the lists of lists of lists...
            return [self._clean(res, depth-1) for res in results]
        if len(results) == 1:
            # Nothing to merge
            return results[0][0]

        out_vals = results[0][0]
        out_times = results[0][1]
        for res in results[1:]:
            start = res[1][0]
            end = res[1][-1]
            if start > out_times[-1]:
                # append after
                out_vals += res[0]
                out_times += res[1]
            elif start == out_times[-1]:
                # append after, bondary repeated
                out_vals += res[0][1:]
                out_times += res[1][1:]
            elif end < out_times[0]:
                # append before
                out_vals = res[0] + out_vals
                out_times = res[1] + out_times
            elif end == out_times[0]:
                # append before, bondary repeated
                out_vals = res[0][:-1] + out_vals
                out_times = res[1][:-1] + out_times
            else:
                # Inside, TODO: skip cache in these case?
                vals = out_vals + res[0]
                time = out_times + res[1]
                idx = np.argsort(time)
                vals = [vals[i] for i in idx]
                time = [time[i] for i in idx]
                _, idx = np.unique(time, return_index=True)
                if len(time) != len(idx):
                    vals = [vals[i] for i in idx]
                    time = [time[i] for i in idx]
                out_vals = vals
                out_times = time
        return out_vals

    def _sort_by_tlist(self, res, tlist, times, depth, check=True):
        """
        'tlist' is the list of user desired times.
        'times' are the computed times.
        If the 'tlist' is not sorted, these could be different.
        Force user desired order.
        """
        if check and np.allclose(tlist, times):
            # tlist sorted, all good
            return res
        if depth:
            # Get to the right depth in the lists of lists of lists...
            return [self._sort_by_tlist(res, tlist, times, depth-1, 0)
                    for res in res_time]
        return [res[times.searchsorted(t)] for t in tlist]

    def _dprop(self, prop, times, tlist, starts_tlist):
        """Used to optain propagator not starting from t0
        input: prop are [U(t, t0) for t in times], tlist, starts_tlist
        return: [U(t1,t2) for t1 in tlist and t2 in starts_tlist]
        U(t1,t2) = U(t1,t0) * U(t2,t0).dag
        """
        new_results = []
        if len(tlist) > len(starts_tlist):
            starts_tlist *= len(tlist)
        elif len(tlist) < len(starts_tlist):
            tlist *= len(starts_tlist)
        for t1, t2 in zip(tlist, starts_tlist):
            idx1 = times.searchsorted(t1)
            idx2 = times.searchsorted(t2)
            prop_t1_t0 = results[idx1]
            prop_t2_t0 = results[idx2]
            prop_t1_t2 = prop_t1_t0 @ prop_t2_t0.T.conj()
            new_results.append(prop_t1_t2)
        return new_results

    def _apply_type(self, results, dtype, dims, depth):
        """ Change the state type: dense, sparse, Qobj.
        """
        # TODO: have from type for faster type conversion
        if depth:
            # Get to the right depth in the lists of lists of lists...
            return [self._apply_type(res, dtype, dims, depth-1)
                    for res in results]
        if dtype == "Qobj":
            # TODO: use fast=...
            results = [Qobj(mat, dims=dims) for mat in results]
        elif dtype == "sparse":
            results = [csr_matrix(mat) for mat in results]
        elif dtype == "dense":
            results = [np.array(mat) for mat in results]
        return results

    def _prop_core(self, t, args):
        """return [(prop(t), t), ...] for each args
        with ts sorted and duplicate removed """
        run_set = []
        run_idx = []
        res = [None] * len(args_sets)
        t = np.unique(np.sort(t))
        for i, arg in enumerate(args_sets):
            prop, times, past, futur = self._cache.get_prop(arg, tlist)
            if past.size > 0:
                # some prop are missing at times before t0.
                prop, t = self._cache.first_after(np.max(past), True)
                run_set.append([arg, np.sort(past), prop, t])
                run_idx.append(i)
            if futur.size > 0:
                prop, t = self._cache.last_before(np.min(futur), True)
                run_set.append([arg, np.sort(futur), prop, t])
                run_idx.append(i)
            if times > 0:
                res[i] = [[prop, times]]
            else:
                res[i] = []

        runs = self._run_sets(run_set)
        for i, run in zip(run_idx, runs):
            res[i].append(run)

        results = self._clean(res, 1)
        return results, t

    def _raw_expect(self, t, state0=None, args=None, e_ops=None):

    def _run_sets(self, sets, prop=False):
        if len(sets) == 1:
            results = [self._run_one(sets)]
        map_kwargs = {'progress_bar': self.progress_bar,
                      'num_cpus': self.options.num_cpus}
        results = self.map_func(self._run_one, sets, **map_kwargs)
        if self.cache_level >= 4 if prop else 2:
            for res, run_data in zip(results, sets):
                set_info, tlist, state0, t0 = *run_data
                self._cache[(*set_info, tlist)] = res
        return results

    def _run_one_alone(self, run_data):
        set_info, tlist, state0, t0 = *run_data
        self.LH_Solver.arguments(set_info[0], state0, self._e_ops0)
        solver = self.get_solver(progress_bar=self.progress_bar,
                                 args=set_info[0])
        return solver(t0, state0, tlist)

    def _run_one(self, run_data):
        set_info, tlist, state0, t0 = *run_data
        self.LH_Solver.arguments(set_info[0], state0, self._e_ops0)
        solver = self.get_solver(args=set_info[0])
        return solver(t0, state0, tlist)


class SESolverV1(Solver):
    """Schrodinger Equation Solver

    """
    def __init__(self, H, args={}, options=None):
        if options is None:
            options = Options()

        super().__init__()
        if isinstance(H, (list, Qobj, QobjEvo)):
            ss = _sesolve_QobjEvo(H, tlist, args, options)
        elif callable(H):
            ss = _sesolve_func_td(H, args, options)
        else:
            raise Exception("Invalid H type")

        self.H = H
        self.ss = ss

        self.dims = self.Hevo.cte.dims
        self._size = self.Hevo.cte.shape[0]

        self._tlist = []
        self._psi = []

        self.options = options
        self._optimization = {"period":0}

        self._args = args
        self._args_n = 0
        self._args_list = [args.copy()]



    def set_initial_value(self, psi0, tlist=[]):
        self.state0 = psi0
        self.dims = psi0.dims
        if tlist:
            self.tlist = tlist

    def optimization(self, period=0):
        self._optimization["period"] = period
        self._cache = False
        raise NotImplementedError

    def run(self, progress_bar=True):
        if progress_bar is True:
            progress_bar = TextProgressBar()

        func, ode_args = self.ss.makefunc(self.ss, self.state0,
                                          self._args, self.options)
        old_store_state = self._options.store_states
        if not self.e_ops:
            self._options.store_states = True

        if self.state0.isket:
            normalize_func = normalize_inplace
            func, ode_args = self.ss.makefunc(self.ss, self.state0,
                                              self._args, self.options)
        else:
            normalize_func = normalize_op_inplace
            func, ode_args = self.ss.makeoper(self.ss, self.state0,
                                              self._args, self.options)

        if not self._options.normalize_output:
            normalize_func = None

        self._e_ops.init(self._tlist)
        self._state_out = self._generic_ode_solve(func, ode_args, self.state0,
                                                  self._tlist, self._e_ops,
                                                  normalize_func, self._options,
                                                  progress_bar)
        self._options.store_states = old_store_state


    def _check_args(self, args):
        pass

    def _check_input(self, psi, args, tlist, level):
        if isinstance(psi, Qobj):
            psi = [psi]
            num_states = 0
        elif not psi:
            psi = [self.psi]
            num_states = 0
        elif isinstance(psi, list):
            if any((not isinstance(ele, Qobj) for ele in psi)):
                raise TypeError("psi must be Qobj")
            num_states = len(psi)
        else:
            raise TypeError("psi must be Qobj")

        if isinstance(args, dict):
            self._check_args(args)
            args = [args]
            num_args = 0
        elif not args:
            args = [self.args]
            num_args = 0
        elif isinstance(args, list):
            for args_set in args:
                if not isinstance(args_set, dict):
                    raise TypeError("args must be dict")
                self._check_args(args_set)


            num_args = len(args)
        else:
            raise TypeError("args must be dict")

        if isinstance(tlist, (int, float)):
            tlist = [self._t0, tlist]
            nt = 0
        elif not tlist:
            tlist = [self._tlist]
            nt = 0
        elif isinstance(args, list):
            if any((not isinstance(ele, (int, float)) for ele in tlist)):
                raise TypeError("tlist must be list of times")
            nt = len(tlist)
        else:
            raise TypeError("tlist must be list of times")

        return (psi, args, tlist), (num_states, num_args, nt)

    def expect(self, e_ops, psis, argss, tlist,
               progress_bar=True, map_func=parallel_map):
        (psi, args, tlist), (num_states, num_args, nt) = \
             self._check_input(psis, args_sets, tlist)

        if progress_bar is True:
         progress_bar = TextProgressBar()
        map_kwargs = {'progress_bar': progress_bar,
                   'num_cpus': self.options.num_cpus}

        if self._with_state:
             state, expect = self._batch_run_ket(states, args_sets,
                                                 map_func, map_kwargs, 0)
        elif (num_states > vec_len):
             state, expect = self._batch_run_prop_ket(states, args_sets,
                                                      map_func, map_kwargs, 0)
        elif num_states >= 2:
             state, expect = self._batch_run_merged_ket(states, args_sets,
                                                        map_func, map_kwargs, 0)
        else:
             state, expect = self._batch_run_ket(states, args_sets,
                                                 map_func, map_kwargs, 0)

        if nt == 0: expect.squeeze(axis=2)
        if num_args == 0: expect.squeeze(axis=1)
        if num_states == 0: expect.squeeze(axis=0)

        return expect

    def evolve(self, psis, args_sets=[], tlist=[],
               progress_bar=True, map_func=parallel_map):
        (psi, args, tlist), (num_states, num_args, nt) = \
            self._check_input(psis, args_sets, tlist)

        if progress_bar is True:
            progress_bar = TextProgressBar()
        map_kwargs = {'progress_bar': progress_bar,
                      'num_cpus': self.options.num_cpus}

        if self._with_state:
            state, expect = self._batch_run_ket(states, args_sets, map_func, map_kwargs, 1)
        elif (num_states > vec_len):
            state, expect = self._batch_run_prop_ket(states, args_sets, map_func, map_kwargs, 1)
        elif num_states >= 2:
            state, expect = self._batch_run_merged_ket(states, args_sets, map_func, map_kwargs, 1)
        else:
            state, expect = self._batch_run_ket(states, args_sets, map_func, map_kwargs, 1)

        states_out = np.empty((len(psis), len(args_sets), len(tlist)),
                              dtype=object)
        for i,j,k in product(range(len(psis)), range(len(args_sets)),
                             range(len(tlist))):
            states_out[i,j,k] = Qobj(data=dense1D_to_fastcsr_ket(state[i,j,k]),
                                     dims=psis[0].dims, fast="mc")
        if nt == 0: states_out.squeeze(axis=2)
        if num_args == 0: states_out.squeeze(axis=1)
        if len(states_out.shape) == 1 and num_states == 0:
            states_out = states_out[0]
        elif num_states == 0:
            states_out.squeeze(axis=0)

        return states_out

    def propagator(self, args, tlist,
                   progress_bar=True, map_func=parallel_map):
        (_, args, tlist), (_, num_args, nt) = \
            self._check_input([], args_sets, tlist)

        self._options.store_states = 1
        computed_state = [qeye(self._size)]
        values = list(product(computed_state, args_sets))

        normalize_func = normalize_inplace
        if not self._options.normalize_output:
            normalize_func = False

        if progress_bar is True:
            progress_bar = TextProgressBar()
        if len(values) == 1:
            map_func = serial_map
        map_kwargs = {'progress_bar': progress_bar,
                      'num_cpus': self.options.num_cpus}

        results = map_func(self._one_run_oper, values,
                           (normalize_func,), **map_kwargs)

        prop = np.empty((num_args, nt), dtype=object)
        for args_n, (states, _) in enumerate(results):
            prop[args_n, :] = [Qobj(data=dense2D_to_fastcsr_fmode(state),
                                   dims=computed_state[0].dims, fast="mc")
                               for state in states]
        if nt == 0: states_out.squeeze(axis=1)
        if len(states_out.shape) == 1 and num_args == 0:
            states_out = states_out[0]
        elif num_args == 0:
            states_out.squeeze(axis=0)
        return prop



    def batch_run(self, states=[], args_sets=[],
                  progress_bar=True, map_func=parallel_map):
        num_states = len(states)
        nt = len(self._tlist)
        if not states:
            states = [self.state0]
        vec_len = self.ss.shape[0]
        state_shape = [state.shape[1] for state in states]
        all_ket = all([n == 1 for n in state_shape])
        all_op = all([n == vec_len for n in state_shape])
        if not (all_ket or all_op):
            raise ValueError("Input state must be all ket or operator")

        num_args = len(args_sets)
        if not args_sets:
            args_sets = [self._args]

        if progress_bar is True:
            progress_bar = TextProgressBar()
        map_kwargs = {'progress_bar': progress_bar,
                      'num_cpus': self.options.num_cpus}

        if all_ket and self.ss.with_state:
            state, expect = self._batch_run_ket(states, args_sets,
                                                map_func, map_kwargs)
        elif all_ket and (num_states > vec_len):
            state, expect = self._batch_run_prop_ket(states, args_sets,
                                                     map_func, map_kwargs)
        elif all_ket and num_states >= 2:
            state, expect = self._batch_run_merged_ket(states, args_sets,
                                                       map_func, map_kwargs)
        elif all_ket:
            state, expect = self._batch_run_ket(states, args_sets,
                                                map_func, map_kwargs)
        elif self.ss.with_state:
            state, expect = self._batch_run_oper(states, args_sets,
                                                 map_func, map_kwargs)
        else:
            state, expect = self._batch_run_prop_oper(states, args_sets,
                                                      map_func, map_kwargs)

        states_out = np.empty((num_states, num_args, nt), dtype=object)
        if all_ket:
            for i,j,k in product(range(num_states), range(num_args), range(nt)):
                states_out[i,j,k] = dense1D_to_fastcsr_ket(state[i,j,k])
        else:
            for i,j,k in product(range(num_states), range(num_args), range(nt)):
                oper = state[i,j,k].reshape((vec_len, vec_len), order="F")
                states_out[i,j,k] = dense2D_to_fastcsr_fmode(oper,
                                                             vec_len, vec_len)

        return states_out, expect

    def _batch_run_ket(self, kets, args_sets, map_func, map_kwargs,
                       store_states):
        num_states = len(kets)
        num_args = len(args_sets)
        nt = len(self._tlist)
        vec_len = self._size
        states_out = np.empty((num_states, num_args, nt, vec_len),
                              dtype=complex)
        expect_out = np.empty((num_states, num_args), dtype=object)
        self._options.store_states = store_states
        values = list(product(kets, args_sets))

        normalize_func = normalize_inplace
        if not self._options.normalize_output:
            normalize_func = False

        results = map_func(self._one_run_ket, values, (normalize_func,),
                           **map_kwargs)

        for i, (state, expect) in enumerate(results):
            args_n, state_n = divmod(i, num_states)
            if self._e_ops:
                expect_out[state_n, args_n] = expect.finish()
            if store_states:
                states_out[state_n, args_n, :, :] = state

        return states_out, expect_out

    def _batch_run_prop_ket(self, kets, args_sets, map_func, map_kwargs,
                       store_states):
        num_states = len(kets)
        num_args = len(args_sets)
        nt = len(self._tlist)
        vec_len = kets[0].shape[0]

        states_out = np.empty((num_states, num_args, nt, vec_len), dtype=complex)
        expect_out = np.empty((num_states, num_args), dtype=object)
        old_store_state = self._options.store_states
        store_states = not bool(self._e_ops) or self._options.store_states
        self._options.store_states = True

        computed_state = [qeye(vec_len)]
        values = list(product(computed_state, args_sets))

        normalize_func = normalize_op_inplace
        if not self._options.normalize_output:
            normalize_func = False

        if len(values) == 1:
            map_func = serial_map
        results = map_func(self._one_run_ket, values, (normalize_func,),
                           **map_kwargs)

        self._options.store_states = old_store_state

        for i, (prop, _) in enumerate(results):
            args_n, state_n = divmod(i, num_states)
            for ket in kets:
                e_op = self._e_ops.copy()
                e_op.init(self._tlist)
                state = np.zeros((nt, vec_len), dtype=np.float)
                for t in self._tlist:
                    state[i,:] = prop[t,:,:] * ket
                    e_op.step(i, state[i,:])
                if self._e_ops:
                    expect_out[state_n, args_n] = e_op.finish()
                if self._options.store_states:
                    states_out[state_n, args_n, :, :] = state
        return states_out, expect_out

    def _batch_run_merged_ket(self, kets, args_sets, map_func, map_kwargs,
                       store_states):
        num_states = len(kets)
        vec_len = kets[0].shape[0]
        num_args = len(args_sets)
        nt = len(self._tlist)

        states_out = np.empty((num_states, num_args, nt, vec_len), dtype=complex)
        expect_out = np.empty((num_states, num_args), dtype=object)
        store_states = not bool(self._e_ops) or self._options.store_states
        old_store_state = self._options.store_states
        self._options.store_states = True
        values = list(product(stack_ket(kets), args_sets))

        normalize_func = normalize_mixed(values[0][0].shape)
        if not self._options.normalize_output:
            normalize_func = False

        if len(values) == 1:
            map_func = serial_map
        results = map_func(self._one_run_ket, values, (normalize_func,),
                           **map_kwargs)

        self._options.store_states = old_store_state

        for args_n, (state, _) in enumerate(results):
            e_ops_ = [self._e_ops.copy() for _ in range(num_states)]
            [e_op.init(self._tlist) for e_op in e_ops_]
            states_out_run = [np.zeros((nt, vec_len), dtype=complex)
                               for _ in range(num_states)]
            for t in range(nt):
                state_t = state[t,:].reshape((num_states, vec_len)).T
                for j in range(num_states):
                    vec = state_t[:,j]
                    e_ops_[j].step(t, vec)
                    if store_states:
                        states_out_run[j][t,:] = vec

            for state_n in range(num_states):
                expect_out[state_n, args_n] = e_ops_[state_n].finish()
                if store_states:
                    states_out[state_n, args_n, :, :] = states_out_run[state_n]

        return states_out, expect_out

    def _batch_run_oper(self, opers, args_sets, map_func, map_kwargs,
                       store_states):
        num_states = len(opers)
        num_args = len(args_sets)
        nt = len(self._tlist)
        vec_len = opers[0].shape[0] * opers.shape[1]

        states_out = np.empty((num_states, num_args, nt, vec_len), dtype=complex)
        expect_out = np.empty((num_states, num_args), dtype=object)
        store_states = not bool(self._e_ops) or self._options.store_states
        old_store_state = self._options.store_states
        self._options.store_states = store_states
        values = list(product(opers, args_sets))

        normalize_func = normalize_inplace
        if not self._options.normalize_output:
            normalize_func = False

        results = map_func(self._one_run_oper, values,
                           (normalize_func,), **map_kwargs)
        self._options.store_states= old_store_state

        for i, (state, expect) in enumerate(results):
            args_n, state_n = divmod(i, num_states)
            if self._e_ops:
                expect_out[state_n, args_n] = expect.finish()
            if store_states:
                states_out[state_n, args_n, :, :] = state

        return states_out, expect_out

    def _batch_run_prop_oper(self, opers, args_sets, map_func, map_kwargs,
                       store_states):
        num_states = len(opers)
        num_args = len(args_sets)
        nt = len(self._tlist)
        vec_len = opers[0].shape[0]
        vec_len_2 = opers[0].shape[0] * opers[0].shape[1]

        states_out = np.empty((num_states, num_args, nt, vec_len_2), dtype=complex)
        expect_out = np.empty((num_states, num_args), dtype=object)
        store_states = not bool(self._e_ops) or self._options.store_states
        old_store_state = self._options.store_states
        self._options.store_states = True
        computed_state = [qeye(vec_len)]
        values = list(product(computed_state, args_sets))

        normalize_func = normalize_op_inplace
        if not self._options.normalize_output:
            normalize_func = False

        if len(values) == 1:
            map_func = serial_map
        results = map_func(self._one_run_ket, values, (normalize_func,),
                           **map_kwargs)

        for args_n, (props, _) in enumerate(results):
            for state_n, oper in enumerate(opers):
                e_op = self._e_ops.copy()
                e_op.init(self._tlist)
                oper_ = oper.full().T
                for t in range(nt):
                    prop = props[t,:].reshape((vec_len, vec_len)).T
                    state = np.conj(prop.T) @ oper_ @ prop
                    e_op.step(t, state.ravel())
                    if store_states:
                        states_out[state_n, args_n, t, :] = state.ravel()
                if self._e_ops:
                    expect_out[state_n, args_n] = e_op.finish()

        self._options.store_states = old_store_state
        return states_out, expect_out

    def _one_run_ket(self, run_data, normalize_func):
        opt = self._options
        state0, args = run_data
        func, ode_args = self.ss.makefunc(self.ss, state0, args, opt)

        if state0.isket:
            e_ops = self._e_ops.copy()
        else:
            e_ops = ExpectOps([])
        state = self._generic_ode_solve(func, ode_args, state0, self._tlist,
                                        e_ops, normalize_func, opt,
                                        BaseProgressBar())
        return state, e_ops

    def _one_run_oper(self, run_data, normalize_func):
        opt = self._options
        state0, args = run_data
        func, ode_args = self.ss.makeoper(self.ss, state0, args, opt)

        e_ops = self._e_ops.copy()
        state = self._generic_ode_solve(func, ode_args, state0, self._tlist,
                                        e_ops, normalize_func, opt,
                                        BaseProgressBar())
        return state, e_ops
