from qutip import destroy, qeye, fock_dm, kraus_to_choi

# Choi matrix for 1 qubit amplitude damping channel with probability p
def AmpDampChoi(p):
    Kraus = [(1-p)**.5*qeye(2),p**.5*destroy(2),p**.5*fock_dm(2,0)]
    return kraus_to_choi(Kraus)

# Choi matrix for identity channel on 1 qubit
IdChoi = kraus_to_choi([qeye(2)])

# These values are found without issue but since they're all CPTP maps they all have dnorm 1
print(IdChoi.dnorm())
print(AmpDampChoi(0).dnorm())
print(AmpDampChoi(0.64).dnorm())


# These trivial diamond norm distances between a channel and itself are also produced
print(IdChoi.dnorm(AmpDampChoi(0)))
print(IdChoi.dnorm(IdChoi))
print(AmpDampChoi(0.5).dnorm(AmpDampChoi(0.5)))


# If a diamond norm distance between two different channels is sought then we see the error
print(IdChoi.dnorm(AmpDampChoi(0.5)))
print(AmpDampChoi(0.4).dnorm(AmpDampChoi(0.5)))