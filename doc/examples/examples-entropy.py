from qutip import *
from pylab import *

a=linspace(0,1,20)
out=zeros(len(a))
for k in range(len(a)):
    x=a[k]*ket2dm(basis(2,0))
    y=(1-a[k])*ket2dm(basis(2,1))
    z=x+y
    rho=Qobj(z)
    out[k]=entropy_vn(rho)

fig=figure(figsize=(6,4))
plot(a,out,lw=1.5)
xlabel('Percentage of excited state')
ylabel('Entropy')
savefig('examples-entropy.png')
close(fig)

