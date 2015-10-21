import scipy


from qutip.mesolve import _generic_ode_solve
from qutip.superoperator import liouvillian, mat2vec
from qutip.settings import debug

if debug:
    import inspect


def _mesolve_const_super(H, E0, tlist, c_op_list, e_ops, args, opt,
                   progress_bar):
    """
    Evolve the super-operator `E0` using an ODE solver, for constant 
    Liouvillian
    """

    if debug:
        print(inspect.stack()[0][3])

    """
    #
    # check initial state
    #
    if isket(rho0):
        # if initial state is a ket and no collapse operator where given,
        # fall back on the unitary schrodinger equation solver
        if len(c_op_list) == 0 and isoper(H):
            return _sesolve_const(H, rho0, tlist, e_ops, args, opt,
                                  progress_bar)

        # Got a wave function as initial state: convert to density matrix.
        rho0 = ket2dm(rho0)
    """

    #
    # check initial value
    #
    if not E0.issuper:
        raise TypeError("Argument 'E0' should be a super-operator")

    #
    # construct liouvillian
    #
    if opt.tidy:
        H = H.tidyup(opt.atol)

    L = liouvillian(H, c_op_list)

    #
    # setup integrator
    #
    initial_vector = mat2vec(E0.full()).ravel()
    # r = scipy.integrate.ode(cy_ode_rhs)
    r = scipy.integrate.ode(_rhs)
    # r.set_f_params(L.data.data, L.data.indices, L.data.indptr)
    # not sure why I need to transpose L here:
    r.set_f_params(scipy.transpose(L.data))
    r.set_integrator('zvode', method=opt.method, order=opt.order,
                     atol=opt.atol, rtol=opt.rtol, nsteps=opt.nsteps,
                     first_step=opt.first_step, min_step=opt.min_step,
                     max_step=opt.max_step)
    r.set_initial_value(initial_vector, tlist[0])

    #
    # call generic ODE code
    #
    return _generic_ode_solve(r, E0, tlist, e_ops, opt, progress_bar)


def _rhs(t,y,data):
    ym = y.reshape(data.shape)
    return (data*ym).flatten()
