function [C, T3, expected2] = clustering(T2, omegaDir, opts)
% for default values, see usage in phalcor.wrapper.
% Author: Tom Shlomo, ACLab BGU, 2020

arguments
    T2 table
    omegaDir (1,:)
    opts.plotFlag
    
    % direct DOA filter
    opts.taumin
    opts.taumax
    
    % weights for metric
    opts.omegaWeight
    opts.tauWeight
    
    % dbscan parameters
    opts.epsilon
    opts.densityThresh
    
    % expected and true positive tolerances
    opts.expected
    opts.tauTol
    opts.omegaTol
end

expected = opts.expected;

T3 = T2(T2.tau >= opts.taumin & T2.tau <= opts.taumax, :);

%% calculate number of neighbors for each point
x = [s2c(T3.omegaRef), T3.tau];
tic;
for i=1:size(T3,1)
    T3.density(i) = nnz( phalcor.omega_tau_metric(x(:,1:3), x(:,4), x(i,1:3), x(i,4), opts.omegaWeight, opts.tauWeight, "epsilon", opts.epsilon, "sqrtFlag", 0, "isCart", 1) <= opts.epsilon^2);
end
fprintf("Density time: %.3f\n", toc);

if opts.plotFlag
    figure("name", "no density filer");
    phalcor.omega_tau_plot(T3.omegaRef, T3.tau, T3.density/max(T3.density), expected);
    set(gca, 'CLim',[0 0.3]);
    
    I = T3.density >= max(T3.density)*opts.densityThresh;
    figure("Name", "with density filter");
    phalcor.omega_tau_plot(T3.omegaRef(I,:), T3.tau(I), T3.density(I), expected);
end

%% DBSCAN clustering
tic;
[c, T3.density2] = mydbscan( x, ...
    @(x, Y) phalcor.omega_tau_metric(Y(:,1:3), Y(:,4), x(:,1:3), x(:,4), opts.omegaWeight,  opts.tauWeight, ...
    "epsilon", opts.epsilon, "sqrtFlag", 0, "isCart", 1)<=opts.epsilon^2,  ...
    ceil(max(T3.density)*opts.densityThresh) );
fprintf("Clustering time: %.3f\n", toc);
I = c>0;
T3.c = c;

if opts.plotFlag
    figure("Name", "clustering, no noise");
    phalcor.omega_tau_plot(T3.omegaRef(I,:), T3.tau(I), c(I), expected);
    colormap(distinguishable_colors(max(c)));
end

%% create clusters table, with averaged estimates
k = max(c);
C = table(zeros(k,2), zeros(k,1), zeros(k,1), 'VariableNames', ["omega", "tau", "count"]);
for i=1:k
    I = c==i;
    C.omega(i,:) = mean_doa(T3.omegaRef(I,:));
    C.tau(i) = mean(T3.tau(I));
    C.count(i) = nnz(I);
end

%% fix estimate of direct sound
[~, i] = min(C.tau);
C.tau(i) = 0;
C.omega(i,:) = omegaDir;

%%
[C, expected2] = phalcor.cluster_stats(C, ...
    expected, opts.tauTol, opts.omegaTol); 

end

