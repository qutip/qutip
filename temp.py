import qutip as qt
import numpy as np
import warnings

# Create two identical pure states
rho1 = qt.fock_dm(2, 0)
rho2 = qt.fock_dm(2, 0)

print("Checking fidelity...")
# This should trigger the warning in your current terminal
fid = qt.fidelity(rho1, rho2)
print(f"Fidelity: {fid}")