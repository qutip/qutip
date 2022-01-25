# Needed to defer evaluating type hints so that we don't need forward
# references and can hide type hint–only imports from runtime usage.
from __future__ import annotations

from base64 import b64decode
from enum import Enum, auto
import os
from tempfile import NamedTemporaryFile
from typing import Union, overload, TYPE_CHECKING
if TYPE_CHECKING:
    from typing_extensions import Literal

try:
    import pyqir_generator as pqg
except ImportError:
    try:
        import pyqir.generator as pqg
    except ImportError as ex:
        raise ImportError("qutip.qip.qir depends on PyQIR") from ex

try:
    import pyqir_parser as pqp
except ImportError:
    try:
        import pyqir.pyqir_parser as pqp
    except ImportError as ex:
        raise ImportError("qutip.qip.qir depends on PyQIR") from ex


from qutip.qip.circuit import Gate, Measurement, QubitCircuit

__all__ = [
    "circuit_to_qir",
    "QirFormat"
]

QREG = "qr"
CREG = "cr"

class QirFormat(Enum):
    """
    Specifies the format used to serialize QIR.
    """
    #: Specifies that QIR should be encoded as LLVM bitcode (typically, files
    #: ending in `.bc`).
    BITCODE = auto()
    #: Specifies that QIR should be encoded as plain text (typicaly, files
    #: ending in `.ll`).
    TEXT = auto()
    #: Specifies that QIR should be encoded as a PyQIR module object.
    MODULE = auto()

    @classmethod
    def ensure(cls, val : Union[Literal["bitcode", "text", "module"], QirFormat]) -> QirFormat:
        """
        Given a value, returns a value ensured to be of type `QirFormat`,
        attempting to convert if needed.
        """
        if isinstance(val, cls):
            return val
        elif isinstance(val, str):
            return cls[val.upper()]

        return cls(val)

# Specify return types for each different format, so that IDE tooling and type
# checkers can resolve the return type based on arguments.
@overload
def circuit_to_qir(circuit : QubitCircuit, format : Union[Literal[QirFormat.BITCODE], Literal["bitcode"]], module_name : str) -> bytes: ...
@overload
def circuit_to_qir(circuit : QubitCircuit, format : Union[Literal[QirFormat.TEXT], Literal["text"]], module_name : str) -> str: ...
@overload
def circuit_to_qir(circuit : QubitCircuit, format : Union[Literal[QirFormat.MODULE], Literal["module"]], module_name : str) -> pqp.QirModule: ...

def circuit_to_qir(circuit, format, module_name = "qutip_circuit"):
    """
    Given a circuit acting on qubits, generates a representation of that
    circuit using Quantum Intermediate Representation (QIR).

    :param circuit: The circuit to be translated to QIR.
    :param format: The QIR serialization to be used. If `"text"`, returns a
        plain-text representation using LLVM IR. If `"bitcode"`, returns a
        dense binary representation ideal for use with other compilation tools.
        If `"module"`, returns a PyQIR module object that can be manipulated
        further before generating QIR.
    :param module_name: The name of the module to be emitted.
    """
    fmt = QirFormat.ensure(format)

    builder = pqg.QirBuilder(module_name)

    builder.add_quantum_register(QREG, circuit.N)
    if circuit.num_cbits:
        builder.add_classical_register(CREG, circuit.num_cbits)

    for op in circuit.gates:
        # If we have a QuTiP gate, then we need to convert it into one of
        # the reserved operation names in the QIR base profile's quantum
        # instruction set (QIS).
        if isinstance(op, Gate):
            # PyQIR does not yet have branching support. Once that feature
            # is added, we'll need to translate control lines into branching
            # like `if creg0 { op(qreg0); }`.
            if op.classical_controls:
                raise NotImplementedError("Classical controls are not yet implemented.")

            # TODO: Validate indices.
            if op.name == "X":
                builder.x(f"{QREG}{op.targets[0]}")
            elif op.name == "Y":
                builder.y(f"{QREG}{op.targets[0]}")
            elif op.name == "Z":
                builder.z(f"{QREG}{op.targets[0]}")
            elif op.name == "S":
                builder.s(f"{QREG}{op.targets[0]}")
            elif op.name == "T":
                builder.t(f"{QREG}{op.targets[0]}")
            elif op.name == "SNOT":
                builder.h(f"{QREG}{op.targets[0]}")
            elif op.name == "CNOT":
                builder.cx(f"{QREG}{op.controls[0]}", f"{QREG}{op.targets[0]}")
            elif op.name == "RX":
                builder.rx(op.control_value, f"{QREG}{op.targets[0]}")
            elif op.name == "RY":
                builder.ry(op.control_value, f"{QREG}{op.targets[0]}")
            elif op.name == "RZ":
                builder.rz(op.control_value, f"{QREG}{op.targets[0]}")
            elif op.name in ("CRZ", "TOFFOLI"):
                raise NotImplementedError(
                    "Decomposition of CRZ and Toffoli gates into base " +
                    "profile instructions is not yet implemented."
                )
            else:
                raise ValueError(
                    f"Gate {op.name} not supported by the QIR base profile, " +
                    "and may require a custom declaration."
                )

        elif isinstance(op, Measurement):
            # TODO: Validate indices.
            if op.name == "Z":
                builder.m(f"{QREG}{op.targets[0]}", f"{CREG}{op.classical_store}")
            else:
                raise ValueError(
                    f"Measurement kind {op.name} not supported by the QIR " +
                    "base profile, and may require a custom declaration."
                )

        else:
            raise NotImplementedError(
                f"Instruction {op} is not implemented in the QIR base " +
                "profile and may require a custom declaration."
            )

    if fmt == QirFormat.TEXT:
        return builder.get_ir_string()
    elif fmt == QirFormat.BITCODE:
        return b64decode(builder.get_bitcode_base64_string())
    elif fmt == QirFormat.MODULE:
        bitcode = b64decode(builder.get_bitcode_base64_string())
        f = NamedTemporaryFile(suffix='.bc', delete=False)
        try:
            f.write(bitcode)
        finally:
            f.close()
        module = pqp.QirModule(f.name)
        try:
            os.unlink(f.name)
        except:
            pass
        return module
    else:
        assert False, "Internal error; should have caught invalid format enum earlier."
