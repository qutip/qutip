from . import Qobj, QobjEvo

__all__ = [
    'isbra', 'isket', 'isoper', 'issuper', 'isoperbra', 'isoperket', 'isherm'
]


def isbra(x: Qobj | QobjEvo) -> bool:
    return isinstance(x, (Qobj, QobjEvo)) and x.type in ['bra', 'scalar']


def isket(x: Qobj | QobjEvo) -> bool:
    return isinstance(x, (Qobj, QobjEvo)) and x.type in ['ket', 'scalar']


def isoper(x: Qobj | QobjEvo) -> bool:
    return isinstance(x, (Qobj, QobjEvo)) and x.type in ['oper', 'scalar']


def isoperbra(x: Qobj | QobjEvo) -> bool:
    return isinstance(x, (Qobj, QobjEvo)) and x.type in ['operator-bra']


def isoperket(x: Qobj | QobjEvo) -> bool:
    return isinstance(x, (Qobj, QobjEvo)) and x.type in ['operator-ket']


def issuper(x: Qobj | QobjEvo) -> bool:
    return isinstance(x, (Qobj, QobjEvo)) and x.type in ['super']


def isherm(x: Qobj) -> bool:
    if not isinstance(x, Qobj):
        raise TypeError(f"Invalid input type, got {type(x)}, exected Qobj")
    return x.isherm
