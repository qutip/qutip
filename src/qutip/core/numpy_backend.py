from ..settings import settings


class NumpyBackend:
    __slots__ = (
        "_qt_np",
        "abs",
        "angle",
        "arccos",
        "array",
        "ceil",
        "conj",
        "cos",
        "cumsum",
        "diff",
        "e",
        "histogram",
        "imag",
        "inf",
        "inner",
        "linalg",
        "log",
        "log2",
        "max",
        "maximum",
        "ones",
        "pi",
        "prod",
        "real",
        "searchsorted",
        "sin",
        "sort",
        "sqrt",
        "sum",
        "where",
        "zeros",
        "zeros_like",
    )

    def _qutip_setting_backend(self, np):
        self._qt_np = np
        for slot in self.__slots__:
            if slot != "_qt_np":
                setattr(self, slot, getattr(np, slot))

    def __getattr__(self, name):
        return getattr(self._qt_np, name)


# Initialize the numpy backend
np = NumpyBackend()
