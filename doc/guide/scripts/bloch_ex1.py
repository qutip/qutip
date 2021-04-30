import numpy as np
import qutip

b = qutip.Bloch()
th = np.linspace(0, 2*np.pi, 20)

xp = np.cos(th)
yp = np.sin(th)
zp = np.zeros(20)

xz = np.zeros(20)
yz = np.sin(th)
zz = np.cos(th)

b.add_points([xp, yp, zp])
b.add_points([xz, yz, zz])
b.show()
