function [T, omegaDir] = post_process_singular_vectors(T, opts)
% given a table T of peaks in rho in all bins, this function estimates the
% DOAs from the singular vectors (variables "u" and "v" in T), filters
% false alarms, and assigns expecteds.
%
% Author: Tom Shlomo, ACLab BGU, 2020


arguments
    T table
    opts.omegaDirHistTol (1,1) double = 10*pi/180
    opts.rhoThresh (1,1) = 0.85 % negative for adaptive
    
    opts.ompTol = sqrt(0.4);
    opts.ompKmax = 3;
    
    opts.plotFlag = 0
    opts.expected
end
expected = opts.expected;


%% omp
% if ompKmax==1, then we can implement it faster than in the general case,
% hence the following serperation.
% also, if ompKmax==1, filterting is performed based on rho, and not based
% on ompError.
if opts.ompKmax == 1
    [rhoRef, T.omegaRef] = sphere_max_abs(T.u.', "normalization", "rho");
    T.rhoRef = rhoRef.';
    T.rho = T.rhoDir.*T.rhoRef;
    
    if adativeRhoThresh
        error("adativeRhoThresh is not supported when ompKmax=1");
    else
        T( T.rho < opts.rhoThresh,  : ) = [];
    end
    fprintf("size after rho filter: %d\n", size(T,1));
else
    omega_grid = sampling_schemes.fliege_maier(29);
    Y_grid = shmat(N,omega_grid);
    T.ompErr = zeros(size(T,1),1);
    omegaRef = cell(size(T,1),1);
    x = cell(size(T,1),1);
    T.ompK = zeros(size(T,1),1);
    H = wbar();
    for i=1:size(T,1)
        u = T.u(i,:).';
    %     v = T.v(i,:).';
    %     u = u - v*(v'*u);
        [omegaRef{i}, err, xtmp] = omp_sh(u, "omega_grid", omega_grid, "Y_grid", Y_grid, "tol", opts.ompTol, "Kmax", opts.ompKmax);
        T.ompErr(i) = err(end);
        x{i} = xtmp(:,end);
        T.ompK(i) = size(omegaRef{i},1);
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
end

if opts.plotFlag
    figure("name", "omega ref");
    omega_tau_plot(T.omegaRef, T.tau, T.rhoDir.*abs(T.rhoRef), expected);
    colorbar;
end
%%
T = assign_expected(T, expected, "plotFlag", opts.plotFlag);

end
