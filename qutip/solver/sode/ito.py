import numpy as np

class ItoNoise:
    def __init__(self, T, dt):
        N = int(np.round(T / dt))
        self.T = T
        self.dt = dt
        self.noise = np.random.randn(N) * dt**0.5

    def dw(self, dt):
        # I(j)
        N = int(np.round(dt /self.dt))
        return self.noise.reshape(-1, N).sum(axis=1)

    def dz(self, dt):
        # I(0, j)
        N = int(np.round(dt /self.dt))
        return self.noise.reshape(-1, N) @ np.arange(N-0.5, 0, -1) * self.dt

class MultiNoise:
    def __init__(self, T, dt, num=1):
        N = int(np.round(T / dt))
        self.T = T
        self.dt = dt
        self.num = num
        self.noise = np.random.randn(N, num) * dt**0.5

    def dw(self, dt):
        N = int(np.round(dt /self.dt))
        return self.noise.reshape(-1, N, self.num).sum(axis=1)

    def dz(self, dt):
        N = int(np.round(dt /self.dt))
        return np.einsum(
            "ijk,j",
            self.noise.reshape(-1, N, self.num),
            np.arange(N-0.5, 0, -1)
        ) * self.dt

    def dW(self, dt):
        N = int(np.round(dt / self.dt))
        noise = self.noise
        if noise.shape[0] % N:
            noise = noise[:-noise.shape[0] % N]
        out = np.empty((self.num, 2), dtype=float)
        out[:, 0] = noise[:N, :].sum(axis=0)
        out[:, 1] = np.arange(N-0.5, 0, -1) @ noise[:N, :] * self.dt
        return out.T
