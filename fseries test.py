from qutip import *



y=fseries(qobj(ones((3,3))))
w=qobj(ones((3,3)))
z=qobj(zeros((3,3)))
x=fseries(create(3))
w=fseries(w)
z=fseries(z)

print y-w
