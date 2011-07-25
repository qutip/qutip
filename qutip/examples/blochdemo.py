from scipy import *
from ..states import *
from ..operators import *
from ..odesolve import *
from ..Bloch import *

def blochdemo():
    from pylab import plot,show
    from matplotlib import mpl,cm #need to import colormap (cm)
    #from matplotlib import mpl,cm #need to import colormap (cm)
    #
    # set up the calculation
    #
    w     = 1.0 * 2 * pi   # qubit angular frequency
    theta = 0.2 * pi       # qubit angle from sigma_z axis (toward sigma_x axis)
    gamma1 = 0.05      # qubit relaxation rate
    gamma2 = 1.0      # qubit dephasing rate

    # Hamiltonian
    sx = sigmax()
    sy = sigmay()
    sz = sigmaz()
    sm = sigmam()
    H = w * (cos(theta) * sz + sin(theta) * sx)
    # collapse operators
    c_op_list = []
    n_th = 0 # zero temperature
    rate = gamma1 * (n_th + 1)
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sm)
    rate = gamma1 * n_th
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sm.dag())
    rate = gamma2
    if rate > 0.0:
        c_op_list.append(sqrt(rate) * sz)
    # evolve and calculate expectation values
    
    # initial state
    a = .5
    psi0 = (a* basis(2,0) + (1-a)*basis(2,1)).unit()
    tlist = linspace(0,3,500)
    
    expt = odesolve(H, psi0, tlist, c_op_list, [sx, sy, sz])
    sx=expt[0]
    sy=expt[1]
    sz=expt[2]

    sphere=Bloch()
    sphere.add_points([sx,sy,sz])
    sphere.point_color=['r']
    sphere.vector_color = ['b']
    sphere.add_vectors([sin(theta),0,cos(theta)])
    sphere.view=[-43,23]
    sphere.show()
    
    sphere.clear() #clear the previous data
    nrm=mpl.colors.Normalize(0,3)#normalize colors to tlist range
    colors=cm.jet(nrm(tlist)) #make list of colors, one for each time in twist
    sphere.point_color=list(colors) #define sphere point colors
    sphere.add_points([sx,sy,sz],'m')#add points as 'single' points
    sphere.point_marker=['o'] #make all markers same 'circle' shape
    sphere.point_size=[25] #same point sizes
    sphere.view=[-7,7] #change viewing angle to see all the colors
    sphere.zlpos=[1.1,-1.2]
    sphere.show()


if __name__=='main()':
	blochdemo()




