
// Name of Experiment: W-state v1

OPENQASM 2.0;
include "qelib1.inc";


qreg q[3];
creg c[3];
gate cH a,b {
h b;
sdg b;
cx a,b;
h b;
t b;
cx a,b;
t b;
h b;
s b;
x b;
s a;
}

ry(1.91063) q[0];
cH q[0],q[1];
ccx q[0],q[1],q[2];
x q[0];
x q[1];
cx q[0],q[1];
