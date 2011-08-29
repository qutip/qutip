%
%
%

format compact

wa  = 1.0 * 2 * pi   % frequency of system a
wb  = 1.0 * 2 * pi   % frequency of system a
wab = 0.1 * 2 * pi   % coupling frequency
%ga = 0.0             % dissipation rate of system a
%gb = 0.0             % dissipation rate of system b
ga = 0.05            % dissipation rate of system a
gb = 0.00            % dissipation rate of system b
Na = 2               % number of states in system a
Nb = 2               % number of states in system b

tlist = 0:350/500:350;

n_runs = 1;

Na_vec = 2:25;

figure(1)
hold on

times = zeros(length(Na_vec), 1);
for n_run=1:n_runs,
    %disp(['run number ', num2str(n_run)])
    n_idx = 1;
    for Na=Na_vec
        %print "using %d states" % (Na * Nb)
        s_idx = 1;

        tic;
        [na, nb] = benchmark_qotoolbox_integrate(Na, Nb, wa, wb, wab, ga, gb, tlist);
        times(n_idx, s_idx) = times(n_idx, s_idx) + toc;
        s_idx = s_idx + 1;

               
        n_idx = n_idx + 1;
        
        
        plot(tlist, na, 'r')
        plot(tlist, nb, 'b')
        
    end
end

times = times / n_runs;

data = [Na_vec'*Nb, times]

%% plot
%f1 = figure(1)
%clf, hold on
%plot(tlist,real(nc), tlist,real(na))
%plot(tlist, ones(size(tlist)) * nc_ss, 'k')
%plot(tlist, ones(size(tlist)) * na_ss, 'k')
%legend('nc', 'na')
%xlabel('Time');
%ylabel('occupation probability');

%% store the data to a file
%res = [tlist',real(nc'), real(na')];

save 'benchmark-qotoolbox-data.dat' data -ascii

