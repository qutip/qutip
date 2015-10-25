"""
This module contains

- Functions to integrate the master equation for k cascaded identical systems.

"""

import scipy as sp


import qutip as qt


from tensorqobj import TensorQobj
from qutip.ui.progressbar import BaseProgressBar


class Simulation:
    """Class of options for

    Attributes
    ----------
    """

    def __init__(self, H_S, L1, L2, times=[], options=None):

        if options is None:
            self.options = qt.Options()
        else:
            self.options = options

        self.H_S = H_S
        self.sysdims = H_S.dims
        self.L1 = L1
        self.L2 = L2
        self.times = times
        self.store_states = self.options.store_states
        # create system identity superoperator
        self.Id = qt.qeye(H_S.shape[0])
        self.Id.dims = self.sysdims
        self.Id = qt.sprepost(self.Id,self.Id)

    def propagator(self, t, tau, notrace=False):
        """
        Compute rho(t)
        """
        k= int(t/tau)+1
        s = t-(k-1)*tau
        G1,E0 = _generator(k,self.H_S,self.L1,self.L2)
        E = _integrate(G1,E0,0.,s,opt=self.options)
        if k>1:
            G2,null = _generator(k-1,self.H_S,self.L1,self.L2)
            G2 = qt.composite(self.Id,G2)
            E = _integrate(G2,E,s,tau,opt=self.options)
        E.dims = E0.dims
        if not notrace:
            E = _genptrace(E, k)
        return qt.Qobj(E)

    def outfieldpropagator(self, blist, tlist, tau, notrace=False):
        """
        Compute <O_n(tn)...O_2(t2)O_1(t1)> for times t1,t2,... and
        O_i = I, b_out, b_out^\dagger, b_loop, b_loop^\dagger
        tlist: list of times t1,..,tn
        blist: corresponding list of operators:
            0: I (nothing)
            1: b_out
            2: b_out^\dagger
            3: b_loop
            4: b_loop^\dagger
        tau: time-delay
        """
        klist = []
        slist = []
        for t in tlist:
            klist.append(int(t/tau)+1)
            slist.append(t-(klist[-1]-1)*tau)
        kmax = max(klist)
        zipped = sorted(zip(slist,klist,blist))
        slist = [s for (s,k,b) in zipped]
        klist = [k for (s,k,b) in zipped]
        blist = [b for (s,k,b) in zipped]

        G1,E0 = _generator(kmax,self.H_S,self.L1,self.L2)
        sprev = 0.
        E = E0
        for i,s in enumerate(slist):
            E = _integrate(G1,E,sprev,s,opt=self.options)
            if klist[i]==1:
                l1 = 0.*qt.Qobj()
            else:
                l1 = _localop(self.L1,klist[i]-1,kmax)
            l2 = _localop(self.L2,klist[i],kmax)
            if blist[i] == 0:
                superop = self.Id
            elif blist[i] == 1:
                superop = qt.spre(l1+l2)
            elif blist[i] == 2:
                superop = qt.spost(l1.dag()+l2.dag())
            elif blist[i] == 3:
                superop = qt.spre(l1)
            elif blist[i] == 4:
                superop = qt.spost(l1.dag())
            else:
                raise ValueError('Allowed values in blist are 0, 1, 2, 3 ' +
                                 'and 4.')
            superop.dims = E.dims
            E = superop*E
            sprev = s
        E = _integrate(G1,E,slist[-1],tau,opt=self.options)

        E.dims = E0.dims
        if not notrace:
            E = _genptrace(E, kmax)
        return qt.Qobj(E)

    def rhot(self,rho0,t,tau):
        """
        Compute rho(t)
        """
        E = self.propagator(t,tau)
        rhovec = qt.operator_to_vector(rho0)
        return qt.vector_to_operator(E*rhovec)


    def outfieldcorr(self,rho0,blist,tlist,tau):
        """
        Compute <O_n(tn)...O_2(t2)O_1(t1)> for times t1,t2,... and
        O_i = b or b^\dagger are output field annihilation/creation operators
        times: list of times t1,..,tn
        blist: corresponding list of 0 for "b" and 1 for "b^\dagger"
        tau: time-delay
        """
        E = self.outfieldpropagator(blist,tlist,tau)
        rhovec = qt.operator_to_vector(rho0)
        return (qt.vector_to_operator(E*rhovec)).tr()


def _localop(op,l,k):
    """
    Create a local operator on the l'th system by tensoring
    with identity operators on all the other k-1 systems
    """
    if l<1 or l>k:
        raise IndexError('index l out of range')
    h = op
    I = qt.qeye(op.shape[0])
    I.dims = op.dims
    for i in range(1,l):
        h = qt.tensor(h,I)
    for i in range(l+1,k+1):
        h = qt.tensor(I,h)
    return h


def _genptrace(E, k):
    E = TensorQobj(E)
    for l in range(k-1):
        E = E.loop()
    return qt.Qobj(E)


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


def _integrate2(L,E0,ti,tf,opt=qt.Options()):
    """
    Basic ode integrator
    """
    opt.store_final_state = True
    if tf > ti:
        sol = qt.mesolve(L, E0, [ti,tf], [], [], {}, opt, BaseProgressBar())
        return sol.final_state
    else: 
        return E0

def _integrate(L,E0,ti,tf,opt=qt.Options()):
    """
    Basic ode integrator
    """
    def _rhs(t,y,L):
        ym = y.reshape(L.shape)
        return (L*ym).flatten()

    from qutip.superoperator import vec2mat
    r = sp.integrate.ode(_rhs)
    r.set_f_params(L.data)
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
