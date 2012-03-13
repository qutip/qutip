#
# Textbook example: groundstate properties of an ultra-strongly coupled atom-cavity system.
# 
#
from qutip import *
from pylab import *
import time
from mpl_toolkits.mplot3d import Axes3D

def compute(N, wc, wa, glist, use_rwa):

    # Pre-compute operators for the hamiltonian
    a  = tensor(destroy(N), qeye(2))
    sm = tensor(qeye(N), destroy(2))
    nc = a.dag() * a
    na = sm.dag() * sm
        
    idx = 0
    na_expt = zeros(shape(glist))
    nc_expt = zeros(shape(glist))
    for g in glist:

        # recalculate the hamiltonian for each value of g
        if use_rwa: 
            H = wc * nc + wa * na + g * (a.dag() * sm + a * sm.dag())
        else:
            H = wc * nc + wa * na + g * (a.dag() + a) * (sm + sm.dag())

        # find the groundstate of the composite system
        spvals,spvecs=sp_eigs(H)
        new_dims  = [H.dims[0], [1] * len(H.dims[0])]
        new_shape = [H.shape[0], 1]
        Qs=array([Qobj(spvecs[:,k],dims=new_dims,shape=new_shape) for k in range(N)])
        Qs=array([ket/ket.norm() for ket in Qs])
        na_expt[idx] = sort(expect(na, Qs))[0]
        nc_expt[idx] = sort(expect(nc, Qs))[0]

        idx += 1

    return nc_expt, na_expt
    

def run():
    #
    # set up the calculation
    #
    wc = 1.0 * 2 * pi   # cavity frequency
    wa = 1.0 * 2 * pi   # atom frequency
    N = 25              # number of cavity fock states
    use_rwa = False     # Set to True to see that non-RWA is necessary in this regime

    glist = linspace(0, 2.5, 75) * 2 * pi # coupling strength vector

    start_time = time.time()
    nc, na = compute(N, wc, wa, glist, use_rwa)
    print 'time elapsed = ' +str(time.time() - start_time) 

    #
    # plot the cavity and atom occupation numbers as a function of 
    #
    figure(1)
    plot(glist/(2*pi), nc,lw=2)
    plot(glist/(2*pi), na,lw=2)
    legend(("Cavity", "Atom excited state"),loc=0)
    xlabel('Coupling strength (g)')
    ylabel('Occupation Number')
    title('# of Photons in the Groundstate')
    show()



run()

