from ..settings import settings


class NumpyBackend:
    def _qutip_setting_backend(self, np):
        self._qt_np = np

    def __getattr__(self, name):
        return getattr(self._qt_np, name)


# Initialize the numpy backend
np = NumpyBackend()
