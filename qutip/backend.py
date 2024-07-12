from .settings import settings

class NumpyBackend:
    @property
    def backend(self):
        return settings.core["numpy_backend"]

    def __getattr__(self, name):
        backend = object.__getattribute__(self, 'backend')
        return getattr(backend, name)

# Initialize the numpy backend
np = NumpyBackend()