from qutip import *

b=Bloch()
xp=[cos(th) for th in linspace(0,2*pi,20)]
yp=[sin(th) for th in linspace(0,2*pi,20)]
zp=zeros(20)
xz=zeros(20)
yz=[sin(th) for th in linspace(0,pi,20)]
zz=[cos(th) for th in linspace(0,pi,20)]
b.add_points([xp,yp,zp])
b.add_points([xz,yz,zz]) 
b.show()
