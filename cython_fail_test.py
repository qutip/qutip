import qutip as q
H = [[q.sigmax(), '1j * ( asdf + 0.0 )']] # Always works
#H = [[q.sigmax(), '1j * ( 0.0 + asdf )']] # Doesn't work
args = {'asdf':1.0}
q.mesolve(H, q.basis(2,0), [0,1], [], [], args=args)