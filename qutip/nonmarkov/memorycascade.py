"""
This module contains

- Functions to integrate the master equation for k cascaded identical systems.

"""

import scipy as sp


import qutip as qt


from tensorqobj import TensorQobj


def propagator(Id,t,tau,H_S,L1,L2,drivefunc=None,options=qt.Options()):
    """
    Compute rho(t)
    """
    k= int(t/tau)+1
    s = t-(k-1)*tau
    dim = H_S.dims[0][0]
    # collapseop for the first system in the cascade
    L2first = _localop(L2,1,k,dim)
    if drivefunc is not None:
        Hdrive = lambda t: 1j*(sp.conj(drivefunc(t))*1.0*L2first
                               -drivefunc(t)*1.0*L2first.dag())
        Ldrive = lambda t: -1j*(qt.spre(Hdrive(t))-qt.spost(Hdrive(t)))
    else:
        Ldrive = None
    G1,E0 = _generator(k,H_S,L1,L2)
    E = _integrate(G1,E0,0.,s,Lfunc=Ldrive,opt=options)
    if k>1:
        G2,null = _generator(k-1,H_S,L1,L2)
        G2 = qt.composite(Id,G2)
        E = _integrate(G2,E,s,tau,Lfunc=Ldrive,opt=options)
    E.dims = E0.dims
    E = TensorQobj(E)
    for l in range(k-1):
        E = E.loop()
    return qt.Qobj(E)


def rhot(rho0,t,tau,H_S,L1,L2,Id,drivefunc=None,options=qt.Options()):
    """
    Compute rho(t)
    """
    E = propagator(Id,t,tau,H_S,L1,L2,drivefunc,options)
    rhovec = qt.operator_to_vector(rho0)
    return qt.vector_to_operator(E*rhovec)


def _localop(op,l,k,dim):
    """
    Create a local operator on the l'th system by tensoring
    with identity operators on all the other k-1 systems
    """
    if l<1 or l>k:
        raise IndexError('index l out of range')
    h = op
    id = qt.qeye(dim)
    for i in range(1,l):
        h = qt.tensor(h,id)
    for i in range(l+1,k+1):
        h = qt.tensor(id,h)
    return h


def _generator(k,H,L1,L2):
    """
    Create the generator for the cascaded chain of k system copies
    """
    # create bare operators
    id = qt.qeye(H.dims[0][0])
    Id = qt.spre(id)*qt.spost(id)
    Hlist = []
    L1list = []
    L2list = []
    for l in range(1,k+1):
        h = H
        l1 = L1
        l2 = L2
        for i in range(1,l):
            h = qt.tensor(h,id)
            l1 = qt.tensor(l1,id)
            l2 = qt.tensor(l2,id)
        for i in range(l+1,k+1):
            h = qt.tensor(id,h)
            l1 = qt.tensor(id,l1)
            l2 = qt.tensor(id,l2)
        Hlist.append(h)
        L1list.append(l1)
        L2list.append(l2)
    # create Lindbladian
    L = qt.Qobj()
    #H0 = 0.5*(Hlist[0]+eps*L2list[0]+np.conj(eps)*L2list[0].dag())
    H0 = 0.5*Hlist[0]
    L0 = L2list[0]
    #L0 = 0.*L2list[0]
    L += qt.liouvillian(H0,[L0])
    E0 = Id
    for l in range(k-1):
        E0 = qt.composite(Id,E0)
        Hl = 0.5*(Hlist[l]+Hlist[l+1]+1j*(L1list[l].dag()*L2list[l+1] 
                                          -L2list[l+1].dag()*L1list[l]))
        Ll = L1list[l] + L2list[l+1]
        L += qt.liouvillian(Hl,[Ll])
    Hk = 0.5*Hlist[k-1]
    Lk = L1list[k-1]
    L += qt.liouvillian(Hk,[Lk])
    E0.dims = L.dims
    # return generator, identity superop E0
    return L,E0


def _integrate(L,E0,ti,tf,Lfunc=None,opt=qt.Options()):
    """
    Basic ode integrator
    """
    def _rhs(t,y,L):
        ym = y.reshape(L.shape)
        return (L*ym).flatten()
    def _rhs_td(t,y,L,Lfunc):
        # Lfunc is time-dependent part of L
        L = L + Lfunc(t).data
        ym = y.reshape(L.shape)
        return (L*ym).flatten()

    from qutip.superoperator import vec2mat
    if Lfunc is None:
        r = sp.integrate.ode(_rhs)
        r.set_f_params(L.data)
    else:
        r = sp.integrate.ode(_rhs_td)
        r.set_f_params(L.data,Lfunc)
    initial_vector = E0.data.toarray().flatten()
    r.set_integrator('zvode', method=opt.method, order=opt.order,
                     atol=opt.atol, rtol=opt.rtol, nsteps=opt.nsteps,
                     first_step=opt.first_step, min_step=opt.min_step,
                     max_step=opt.max_step)
    r.set_initial_value(initial_vector, ti)
    r.integrate(tf)
    if not r.successful():
        raise Exception("ODE integration error: Try to increase "
                        "the allowed number of substeps by increasing "
                        "the nsteps parameter in the Options class.")

    return qt.Qobj(vec2mat(r.y)).trans()



