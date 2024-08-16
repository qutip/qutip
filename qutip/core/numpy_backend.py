from ..settings import settings

class NumpyBackend:
    def _qutip_setting_backend(self, np):
        self.np = np

    def __getattr__(self, name):
        return getattr(self.np, name)


# Initialize the numpy backend
np = NumpyBackend()
