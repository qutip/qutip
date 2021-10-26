include "qelib1.inc";

qreg q[2];
creg c[4];
gate cu1fixed (a) c,t {
u1 (-a) t;
cx c,t;
u1 (a) t;
cx c,t;

gate cu c,t {
cu1fixed (3*pi/8) c,t;
}

h q[0];
