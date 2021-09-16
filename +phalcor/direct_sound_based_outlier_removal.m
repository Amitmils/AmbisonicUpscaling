function [T, omegaDir] = direct_sound_based_outlier_removal(T, opts)
% Author: Tom Shlomo, ACLab BGU, 2020

arguments
    T table
    opts.omegaDirHistTol (1,1) double = 10*pi/180
    opts.rhoThresh (1,1) = 0.85 % negative for adaptive
    
    opts.expected table
    opts.plotFlag = 0
end
%% setup
fprintf("initial size: %d\n", size(T,1));
expected = opts.expected;

omegaDirExp = expected.omega(expected.delay==0, :);
if isempty(omegaDirExp)
    omegaDirExp = [nan nan];
end

%% filter by rhoDir
% rhoDir thresh applying Newton's method
T( T.rhoDir < opts.rhoThresh-0.05,  : ) = [];
fprintf("size after first rhoDir filter: %d\n", size(T,1));

% improve omegaDir and rhoDir using Newton's method
[rhoDir, T.omegaDir] = sphere_max_abs(T.v.', "normalization", "rho");
T.rhoDir = rhoDir.';
T.omegaDirErr = angle_between(T.omegaDir, omegaDirExp);
fprintf("size after final rhoDir filter: %d\n", size(T,1));

%% Estimate omega dir
if opts.plotFlag
    figure("name", "omegaDir")
    tl = tiledlayout(2,1,"Padding", "compact", "TileSpacing", "compact");
    nexttile(tl);
end

[hist_omegaDir, bins] = sphere_hist(T.omegaDir, 'bins', sampling_schemes.fliege_maier(29), 'tol', opts.omegaDirHistTol, 'type', "overlap", "plotFlag", opts.plotFlag);
s = sampling_schemes.stats(bins);
disp(s);
[~,k] = max(hist_omegaDir);
omegaDir = mean_doa( T.omegaDir( angle_between( T.omegaDir, bins(k,:) ) <= opts.omegaDirHistTol , :) );
fprintf("omegaDir error: %.2f deg\n", angle_between(omegaDir, omegaDirExp)*180/pi);


%% filter by omega dir
I = angle_between( T.omegaDir, omegaDir ) <= opts.omegaDirHistTol;

if opts.plotFlag
    hold on;
    hammer.plot3( omegaDirExp, [], max(hist_omegaDir), 'mo', "MarkerSize", 10);
    mylegend("hist", "$\Omega_0$");
    nexttile(tl);
    hammer.plot(T.omegaDir(~I,:), [], '.');
    hold on;
    hammer.plot(T.omegaDir(I,:), [], '.');
    hammer.plot3(omegaDirExp, [], 1,'mo', "MarkerSize", 10);
    hammer.plot3(omegaDir, [], 1, 'rx', "MarkerSize", 10);
    mylegend("outside", "inside", "$\Omega_0$", "$\hat{\Omega}_0$");

end

T = T(I,:);
fprintf("size after omegaDir filter: %d\n", size(T,1));


end

