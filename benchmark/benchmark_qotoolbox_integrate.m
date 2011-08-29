%
%
%
function [count_a, count_b] = benchmark_qotoolbox_integrate(Na, Nb, wa, wb, wab, ga, gb, tlist)

    disp(['BM with ', num2str(Na), ' x ', num2str(Nb)]) 


    % hamiltonian
    a = tensor(destroy(Na), identity(Nb));
    b = tensor(identity(Na), destroy(Nb));
    na = a' * a;
    nb = b' * b;
    H = wa * na + wb * nb + wab * (a' * b + a * b');

    % initial state
    psi0 = tensor(basis(Na, Na), basis(Nb, Nb-1));

    % collapse operators
    C1 = sqrt(ga) * a;
    C2 = sqrt(gb) * b;
    C1dC1 = C1'*C1;
    C2dC2 = C2'*C2;
    LH = -i *(spre(H) - spost(H));
    L1 = spre(C1)*spost(C1') - 0.5*spre(C1dC1)-0.5*spost(C1dC1);
    L2 = spre(C2)*spost(C2') - 0.5*spre(C2dC2)-0.5*spost(C2dC2);
    L = LH + L1 + L2;

    rhoES = ode2es(L, psi0 * psi0');

    count_a = esval(expect(na, rhoES), tlist);
    count_b = esval(expect(nb, rhoES), tlist);

    
