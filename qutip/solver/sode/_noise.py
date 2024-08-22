import numpy as np

__all__ = ["Wiener", "PreSetWiener"]


class Wiener:
    """
    Wiener process.
    """
    def __init__(self, t0, dt, generator, shape):
        self.t0 = t0
        self.dt = dt
        self.shape = shape
        self.generator = generator
        self.noise = np.zeros((0,) + shape, dtype=float)
        self.last_W = np.zeros(shape[-1], dtype=float)
        self.idx_last_0 = 0

    def _extend(self, idx):
        N_new_vals = idx - self.noise.shape[0]
        dW = self.generator.normal(
            0, np.sqrt(self.dt), size=(N_new_vals,) + self.shape
        )
        self.noise = np.concatenate((self.noise, dW), axis=0)

    def dW(self, t, N):
        # Find the index of t.
        # Rounded to the closest step, but only multiple of dt are expected.
        idx0 = round((t - self.t0) / self.dt)
        if idx0 + N - 1 >= self.noise.shape[0]:
            self._extend(idx0 + N)
        return self.noise[idx0:idx0 + N, :, :]

    def __call__(self, t):
        """
        Return the Wiener process at the closest ``dt`` step to ``t``.
        """
        # The Wiener process is not used directly in the evolution, so it's
        # less optimized than the ``dW`` method.

        # Find the index of t.
        # Rounded to the closest step, but only multiple of dt are expected.
        idx = round((t - self.t0) / self.dt)
        if idx >= self.noise.shape[0]:
            self._extend(idx + 1)

        if self.idx_last_0 > idx:
            # Before last call, reseting
            self.idx_last_0 = 0
            self.last_W = np.zeros(self.shape[-1], dtype=float)

        self.last_W = self.last_W + np.sum(
            self.noise[self.idx_last_0:idx+1, 0, :], axis=0
        )

        self.idx_last_0 = idx
        return self.last_W


class PreSetWiener(Wiener):
    def __init__(self, noise, tlist, n_sc_ops, heterodyne, is_measurement):
        if heterodyne:
            if noise.shape != (n_sc_ops/2, 2, len(tlist)-1):
                raise ValueError(
                    "Noise is not of the expected shape: "
                    f"{(n_sc_ops/2, 2, len(tlist)-1)}"
                )
            noise = np.reshape(noise, (n_sc_ops, len(tlist)-1), order="C")
        else:
            if noise.shape != (n_sc_ops, len(tlist)-1):
                raise ValueError(
                    "Noise is not of the expected shape: "
                    f"{(n_sc_ops, len(tlist)-1)}"
                )

        self.t0 = tlist[0]
        self.dt = tlist[1] - tlist[0]
        self.shape = noise.shape[1:]
        self.noise = noise.T[:, np.newaxis, :].copy()
        self.last_W = np.zeros(self.shape[-1], dtype=float)
        self.idx_last_0 = 0
        self.is_measurement = is_measurement
        if self.is_measurement:
            # Measurements is scaled as <M> + dW / dt
            self.noise *= self.dt
            if heterodyne:
                self.noise /= 2**0.5

    def _extend(self, N):
        raise ValueError("Requested time is outside the integration range.")


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
