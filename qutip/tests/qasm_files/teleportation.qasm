OPENQASM 2.0;
include "qelib1.inc";

qreg q[3];
creg c0[1];
creg c1[1];

h q[1];
cx q[1], q[2];
cx q[0], q[1];
h q[0];

measure q[0] -> c1[0]
measure q[1] -> c0[0]

if(c0==1) x q[2]
if(c1==1) z q[2]
