from qutip import *
from pylab import *

def probevolve(E,kappa,gamma,g,wc,w0,wl,N,tlist):

    ida    = qeye(N)
    idatom = qeye(2)

    # Define cavity field and atomic operators
    a  = tensor(destroy(N),idatom)
    sm = tensor(ida,sigmam())

    # Hamiltonian
    H = (w0-wl)*sm.dag()*sm + (wc-wl)*a.dag()*a + 1j*g*(a.dag()*sm - sm.dag()*a) + E*(a.dag()+a)

    #collapse operators
    C1=sqrt(2*kappa)*a
    C2=sqrt(gamma)*sm
    C1dC1=C1.dag()*C1
    C2dC2=C2.dag()*C2

    #intial state
    psi0 = tensor(basis(N,0),basis(2,1))
    rho0 = psi0 * trans(psi0);

    # Calculate the Liouvillian
    c_op_list = [C1, C2]
    n_op      = len(c_op_list)
    L = -1j * (spre(H) - spost(H))
    for m in range(0, n_op):
        cdc = c_op_list[m].dag() * c_op_list[m]
        L += spre(c_op_list[m])*spost(c_op_list[m].dag())-0.5*spre(cdc)-0.5*spost(cdc)

       
    # Calculate solution as an exponential series
    start_time=time.time()
    rhoES = ode2es(L,rho0);
    print 'time elapsed (ode2es) = ' +str(time.time()-start_time) 
    
    # Calculate expectation values
    start_time=time.time()
    count1  = esval(scalar_expect(C1dC1,rhoES),tlist);
    count2  = esval(scalar_expect(C2dC2,rhoES),tlist);
    infield = esval(scalar_expect(a,rhoES),tlist);
    print 'time elapsed (esval) = ' +str(time.time()-start_time) 

    return count1, count2, infield


#-----------------------------------------------------------------------------
# 
#--------------------------------------------------------------------------
kappa = 2; 
gamma = 0.2;
g  = 1; 
wc = 0; 
w0 = 0; 
wl = 0;
N  = 4; 
E  = 0.5;

tlist = linspace(0,10,200);

start_time=time.time()
[count1,count2,infield] = probevolve(E,kappa,gamma,g,wc,w0,wl,N,tlist);
print 'time elapsed = ' +str(time.time()-start_time) 

plot(tlist,real(count1))
plot(tlist,real(count2))
xlabel('Time')
ylabel('Transmitted Intensity and Spontaneous Emission')
show()

