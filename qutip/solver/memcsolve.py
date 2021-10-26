def _prob_memcsolve(state):
    return _data.norm.trace(unstack_columns(state))

class MeMcSolver(McSolver):
    """
    ... TODO

    McSolve with jump on some c_ops, but not all.
    Can replace photocurrent_mesolve: give the same result up to `dt`,
    but more efficient.
    """
    def __init__(self, H, c_ops, sc_ops, e_ops=None, options=None,
                 times=None, args=None, feedback_args=None,
                 _safe_mode=False):
        _time_start = time()
        self.stats = {}
        e_ops = e_ops or []
        options = options or SolverOptions()
        args = args or {}
        feedback_args = feedback_args or {}
        if not isinstance(options, SolverOptions):
            raise ValueError("options must be an instance of "
                             "qutip.solver.SolverOptions")

        self._safe_mode = _safe_mode
        self._super = True
        self._state = None
        self._state0 = None
        self._t = 0
        self._seeds = []

        self.e_ops = e_ops
        self.options = options

        H = _to_qevo(H, args, times)
        c_evos = []
        for op in c_ops:
            c_evos.append(_to_qevo(op, args, times))
        sc_evos = []
        for op in sc_ops:
            sc_evos.append(_to_qevo(op, args, times))

        ns_evos = [spre(op._cdc()) + spost(op._cdc()) for op in sc_evos]
        n_evos = [spre(op) * spost(op.dag()) for op in sc_evos]
        self._system = liouvillian(H, c_evos)
        for n_evo in ns_evos:
            self._system -= 0.5 * n_evo
        self.c_ops = n_evos
        self._evolver = McEvolver(self, self.c_ops, n_evos,
                                  options, args, feedback_args)
        self._evolver.norm_func = _prob_memcsolve
        self._evolver.prob_func = _prob_memcsolve
        self.stats["preparation time"] = time() - _time_start
        self.stats['solver'] = "MonteCarlo Master Equation Evolution"
        self.stats['num_collapse'] = len(c_ops)

    def _prepare_state(self, state):
        if isket(state):
            state = ket2dm(state)
        self._state_dims = state.dims
        self._state_shape = state.shape
        self._state_type = state.type
        self._state_qobj = state

        # TODO: with #1420, it should be changed to `in to._str2type`
        if self.options.ode["State_data_type"] in to.dtypes:
            state = state.to(self.options.ode["State_data_type"])
        self._state0 = stack_columns(state.data)
        return self._state0

    def _restore_state(self, state, copy=True):
        dm = unstack_columns(state)
        norm = 1/_data.norm.trace(dm)
        dm = _data.mul(dm, norm)
        return Qobj(dm,
                    dims=self._state_dims,
                    type=self._state_type,
                    copy=copy)
