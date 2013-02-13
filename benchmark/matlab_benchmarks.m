function []=matlab_benchmarks()
%Create array to hold test results
test_results=zeros(1,2);


%perform basic operator algebra to construct JC-model
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
xlswrite('matlab_benchmarks',test_results);