from qutip import *



y=fseries(Qobj(ones((3,3))))
w=Qobj(ones((3,3)))
z=Qobj(zeros((3,3)))
x=fseries(create(3))
w=fseries(w)
z=fseries(z)

print y-w
