from qutip import *
from pylab import *

#
# plot angular wave function for l=3
#
phi=linspace(0,2*pi,90)
theta=linspace(0,pi,45)

c2=basis(7,4) #2l+1

y=orbital(theta,phi,c2)

sphereplot(theta,phi,y)
savefig('examples-angular-sphereplot.png')

#
# approximation to a direction eigenket
#
L=2

theta = linspace(0,   pi, 180)
phi   = linspace(0, 2*pi,  30)

lmax  = 10

psi_list = []

for l in range(0,lmax+1):
    psi_list.append(sqrt((2*l + 1)/(4*pi)) * basis(2*l + 1, l))
psi = orbital(theta, phi, psi_list)

sphereplot(theta, phi, psi)
savefig('examples-angular-direction.png')
