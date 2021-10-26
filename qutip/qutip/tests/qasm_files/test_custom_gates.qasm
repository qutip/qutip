OPENQASM 2.0;
include "qelib1.inc";

qreg q[2];

gate u3alt(theta, gamma, alpha) a{
rz(alpha) a;
ry(theta) a;
rz(gamma) a;
}

gate testcu(theta, gamma, alpha) a, b {
rz(-gamma) b;
u3alt(theta, gamma, alpha) b;
rz(-alpha) b;
cx a, b;
}

u3alt(pi/4, -pi/4, pi/4) q[1];
U(pi/4, -pi/4, pi/4) q[1];

testcu(pi/2, pi/2, pi/2) q[0], q[1];
