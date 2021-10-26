// Implementation of Deutsch algorithm with two qubits for f(x)=x
OPENQASM 2.0;
include "qelib1.inc";

qreg q[5];
creg c[5];

h q[3];
post q[3];
measure q[3] -> c[3];
