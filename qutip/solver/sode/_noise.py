import numpy as np

__all__ = ["Wiener"]


class Wiener:
    """
    Wiener process.
    """
    def __init__(self, t0, dt, generator, shape):
        self.t0 = t0
        self.dt = dt
        self.generator = generator
        self.t_end = t0
        self.shape = shape
        self.process = np.zeros((1,) + shape, dtype=float)

    def _extend(self, t):
        N_new_vals = int((t - self.t_end + self.dt*0.01) // self.dt)
        dW = self.generator.normal(
            0, np.sqrt(self.dt), size=(N_new_vals,) + self.shape
        )
        W = self.process[-1, :, :] + np.cumsum(dW, axis=0)
        self.process = np.concatenate((self.process, W), axis=0)
        self.t_end = self.t0 + (self.process.shape[0] - 1) * self.dt

    def dW(self, t, N):
        if t + N * self.dt > self.t_end:
            self._extend(t + N * self.dt)
        idx0 = int((t - self.t0 + self.dt * 0.01) // self.dt)
        return np.diff(self.process[idx0:idx0 + N + 1, :, :], axis=0)

    def __call__(self, t):
        if t > self.t_end:
            self._extend(t)
        idx = int((t - self.t0 + self.dt * 0.01) // self.dt)
        return self.process[idx, 0, :]


class _Noise:
    """
    Wiener process generator used for tests.
    """

    def __init__(self, T, dt, num=1):
        N = int(np.round(T / dt))
        self.T = T
        self.dt = dt
        self.num = num
        self.noise = np.random.randn(N, num) * dt**0.5

    def dw(self, dt):
        """
        Ito integral I(i).
        """
        N = int(np.round(dt / self.dt))
        return self.noise.reshape([-1, N, self.num]).sum(axis=1)

    def dz(self, dt):
        """
        Ito integral I(0, i).
        """
        N = int(np.round(dt / self.dt))
        return (
            np.einsum(
                "ijk,j->ik",
                self.noise.reshape([-1, N, self.num]),
                np.arange(N - 0.5, 0, -1),
            )
            * self.dt
        )

    def dW(self, dt):
        """
        Noise used for Ito-Taylor integrators of order up to 1.5.
        """
        N = int(np.round(dt / self.dt))
        noise = self.noise.copy()
        if noise.shape[0] % N:
            noise = noise[: -(noise.shape[0] % N)]
        out = np.empty((noise.shape[0] // N, 2, self.num), dtype=float)
        out[:, 0, :] = noise.reshape([-1, N, self.num]).sum(axis=1)
        out[:, 1, :] = (
            np.einsum(
                "ijk,j->ik",
                self.noise.reshape([-1, N, self.num]),
                np.arange(N - 0.5, 0, -1),
            )
            * self.dt
        )
        return out
