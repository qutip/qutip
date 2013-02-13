function []=matlab_benchmarks()
%Create array to hold test results
test_results=zeros(1,4);


%test #1
%perform basic operator algebra to construct JC-Hamiltonian
Nc=20;
wc = 1.0 * 2 * pi;   
wa = 1.0 * 2 * pi;
g  = 0.05 * 2 * pi;
tic;
a=tensor(destroy(Nc),identity(2));
%create spin-operators
sm=tensor(identity(Nc),sigmam());
%build Jaynes-Cummings Hamiltonian
H=wc * a' * a + wa * sm' * sm + g * (a' + a) * (sm + sm');
time=toc;
test_results(1,1)=time;

%test #2
%matrix exponentiation to construct squeezed state and coherent state
N=25;
alpha=2+2i;
sp=1.25i;
tic;
a=destroy(N);
grnd=basis(N);
D_oper=expm(alpha*a'-conj(alpha)*a);
S_oper=expm((1/2.0)*conj(sp)*a^2-(1/2.0)*sp*(a')^2);
coh_state=D_oper*grnd;
sqz_state=S_oper*grnd;
time=toc;
test_results(1,2)=time;

%test #3
%cavity+qubit steady state
N=5;kappa = 2; gamma = 0.2; g = 1;
wc = 0; w0 = 0; wl = 0; E = 0.5;
tic;
ida = identity(N); idatom = identity(2); 
a  = tensor(destroy(N),idatom);
sm = tensor(ida,sigmam);
H = (w0-wl)*sm'*sm + (wc-wl)*a'*a + i*g*(a'*sm - sm'*a) + E*(a'+a);
C1    = sqrt(2*kappa)*a;% Collapse operators
C2    = sqrt(gamma)*sm;
C1dC1 = C1'*C1;
C2dC2 = C2'*C2;
LH = -i * (spre(H) - spost(H)); % Calculate the Liouvillian
L1 = spre(C1)*spost(C1')-0.5*spre(C1dC1)-0.5*spost(C1dC1);
L2 = spre(C2)*spost(C2')-0.5*spre(C2dC2)-0.5*spost(C2dC2);
L  = LH+L1+L2;
rhoss = steady(L);% Find steady state
time=toc;
test_results(1,3)=time;

%test #4
%cavity+qubit master equation
kappa = 2; gamma = 0.2; g = 1;
wc = 0; w0 = 0; wl = 0; E = 0.5;
N = 10;
tlist = linspace(0,10,200);
tic;
ida = identity(N); idatom = identity(2); 
a  = tensor(destroy(N),idatom);
sm = tensor(ida,sigmam);
H = (w0-wl)*sm'*sm + (wc-wl)*a'*a + i*g*(a'*sm - sm'*a) + E*(a'+a);
C1  = sqrt(2*kappa)*a;
C2  = sqrt(gamma)*sm;
C1dC1 = C1'*C1;
C2dC2 = C2'*C2;
LH = -i * (spre(H) - spost(H)); 
L1 = spre(C1)*spost(C1')-0.5*spre(C1dC1)-0.5*spost(C1dC1);
L2 = spre(C2)*spost(C2')-0.5*spre(C2dC2)-0.5*spost(C2dC2);
L = LH+L1+L2;
psi0 = tensor(basis(N,1),basis(2,2));
rho0 = psi0 * psi0';
ode2file('file1.dat',L,rho0,tlist);
odesolve('file1.dat','file2.dat');
fid = fopen('file2.dat','rb');
rho = qoread(fid,dims(rho0),size(tlist));
time=toc;
test_results(1,4)=time;


%test #5
%cavity+qubit monte carlo




xlswrite('matlab_benchmarks',test_results);