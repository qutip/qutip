import numpy as np
from itertools import product
from qutip.core import data as _data
import qutip

def compute_step(system, method, state, ts, noise, **kw):
    import qutip.solver.sode._sode as _sode
    stepper = getattr(_sode, method)(system, **kw)
    out = []
    for t in ts:
        out.append(stepper.run(0, state.copy(), t, noise.dW(t), 1))
    return out


def get_error_order(system, method):
    num_runs = 10
    ts = [
        0.000001, 0.000002, 0.000005, 0.00001, 0.00002,  0.00005,
        0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1,
    ]
    state = rand_ket(system.dims[0]).data
    err = np.zeros(len(ts), dtype=float)
    for _ in range(num_runs):
        noise = MultiNoise(0.1, 0.000001)
        out = compute_step(system, method, state, ts, noise)

        for i, (o, t) in enumerate(zip(out, ts)):
            got = o.to_array()[0,0]
            target = system.analytic(t, noise.dw(t)[0,0])
            err[i] += (np.abs(got - target))
    err /= num_runs
    return np.polyfit(np.log(ts), np.log(err), 1)
