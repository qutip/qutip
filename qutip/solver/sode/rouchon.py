

import qutip as qt
import numpy as np
from qutip.core import data as _data

class Rouchon:
    def __init__(self, rhs, options):
        self.H = rhs.H
        if self.H.issuper:
            raise TypeError("...")
        dtype = type(self.H(0).data)
        self.c_ops = rhs.c_ops
        self.sc_ops = rhs.sc_ops
        self.cpcds = [op + op.dag() for op in self.sc_ops]
        self.M = (
            - 1j * self.H
            - sum(op.dag() @ op for op in self.c_ops) * 0.5
            - sum(op.dag() @ op for op in self.sc_ops) * 0.5
        )
        self.num_collapses = len(self.sc_ops)
        self.scc = [
            [self.sc_ops[i] @ self.sc_ops[j] for i in range(j+1)]
            for j in range(self.num_collapses)
        ]

        self.id = _data.identity[dtype](self.H.shape[0])

    def step(self, t, state, dt, dW):
        # Same output as old rouchon up to nuerical error 1e-16
        # But  7x slower
        dy = [
            op.expect_data(t, state) * dt + dw
            for op, dw in zip(self.cpcds, dW[:, 0])
        ]
        M = _data.add(self.id, self.M._call(t), dt)
        for i in range(self.num_collapses):
            M = _data.add(M, self.sc_ops[i]._call(t), dy[i])
            M = _data.add(M, self.scc[i][i]._call(t), (dy[i]**2-dt)/2)
            for j in range(i):
                M = _data.add(M, self.scc[i][j]._call(t), dy[i]*dy[j])
        out = M @ state @ M.adjoint()
        for cop in self.c_ops:
            op = cop._call(t)
            out += op @ state @ op.adjoint() * dt
        return out / _data.trace(out)
