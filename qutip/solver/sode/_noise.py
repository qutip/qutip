import numpy as np

__all__ = []


class _Noise:
    """
    Weiner process generator used for tests.
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
        return self.noise.reshape(-1, N, self.num).sum(axis=1)

    def dz(self, dt):
        """
        Ito integral I(0, i).
        """
        N = int(np.round(dt / self.dt))
        return (
            np.einsum(
                "ijk,j->ik",
                self.noise.reshape(-1, N, self.num),
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
        out[:, 0, :] = noise.reshape(-1, N, self.num).sum(axis=1)
        out[:, 1, :] = (
            np.einsum(
                "ijk,j->ik",
                self.noise.reshape(-1, N, self.num),
                np.arange(N - 0.5, 0, -1),
            )
            * self.dt
        )
        return out
