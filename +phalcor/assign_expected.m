function T = assign_expected(T, expected, opts)
arguments
    T table
    expected table
    opts.omegaTol = 10*pi/180;
    opts.tauTol = 0.5*1e-3;
    opts.plotFlag  = false
end

% Author: Tom Shlomo, ACLab BGU, 2020

K = size(T,1);
T.idxExp = nan(K,1);
T.omegaRefErr = inf(K,1);
T.omegaDirErr = angle_between(T.omegaDir, expected.omega(1,:));
T.omegaDirExp = repmat(expected.omega(1,:), size(T,1), 1);

dtau = abs(T.tau - expected.delay');
domega = angle_between(T.omegaRef, expected.omega);
I = dtau <= opts.tauTol & domega <= opts.omegaTol;
domega(~I) = inf;
[T.omegaRefErr, T.idxExp] = min(domega, [], 2);
T.isFA = isinf(T.omegaRefErr);
T.idxExp(T.isFA) = 0;
T.omegaRefExp = nan(K,2);
T.omegaRefExp(~T.isFA,:) = expected.omega(T.idxExp(~T.isFA), :);
T.tauExp = nan(K,1);
T.tauExp(~T.isFA) = expected.delay(T.idxExp(~T.isFA));
T.tauErr = T.tau - T.tauExp;
T.tauErr(T.isFA) = inf;

if opts.plotFlag
    figure("name", "assign expected");
    h = phalcor.omega_tau_plot(T.omegaRef, T.tau, T.idxExp-1, expected);
    k = size(expected,1);
    colormap(linspecer(k+1));
    h.Parent.CLim = [-1.5 k-0.5];
    cbar = colorbar;
    cbar.Ticks = 0:k;
    cbar.Label.String = "expected ID";
end

end