function T = omp(T, opts)

arguments
    T table
    opts.ompTol = sqrt(0.4);
    opts.ompKmax = 3;
    
    opts.plotFlag = 0
    opts.expected
end

% Author: Tom Shlomo, ACLab BGU, 2020

expected = opts.expected;
N = sqrt(size(T.u, 2))-1;

omega_grid = sampling_schemes.fliege_maier(29);
Y_grid = shmat(N,omega_grid);
T.ompErr = zeros(size(T,1),1);
omegaRef = cell(size(T,1),1);
x = cell(size(T,1),1);
T.ompK = zeros(size(T,1),1);
H = wbar();
for i=1:size(T,1)
    u = T.u(i,:).';
    [omegaRef{i}, err, xtmp] = omp_sh(u, "omega_grid", omega_grid, "Y_grid", Y_grid, "tol", opts.ompTol, "Kmax", opts.ompKmax);
    T.ompErr(i) = err(end);
    x{i} = xtmp(:,end);
    T.ompK(i) = size(omegaRef{i},1);
    
    % update waitbar every 1000 iterations
    if mod(i,1000)==1 || i==size(T,1)
        wbar(i, size(T,1), H);
    end
end

I = T.ompErr <= opts.ompTol;

if opts.plotFlag
    figure("name", "omp");
    tiledlayout(2,1);
    nexttile();
    histogram(T.ompErr, "Normalization", "probability");
    xline(opts.ompTol, "r");
    nexttile();
    histogram(T.ompK, "Normalization", "probability");
    hold on;
    histogram(T.ompK(I), "Normalization", "probability");
end
T = T(I,:);
fprintf("size after ompErr filter: %d\n", size(T,1));

% expand table
T = T( repelem(1:size(T,1), T.ompK), :);
fprintf("size after expansion: %d\n", size(T,1));

omegaRef = omegaRef(I);
x = x(I);
T.omegaRef = vertcat(omegaRef{:});
T.rhoRef = vertcat(x{:});


if opts.plotFlag
    figure("name", "omega ref");
    phalcor.omega_tau_plot(T.omegaRef, T.tau, T.rhoDir.*abs(T.rhoRef), expected);
    colorbar;
end
%%
T = phalcor.assign_expected(T, expected, "plotFlag", opts.plotFlag);

end

