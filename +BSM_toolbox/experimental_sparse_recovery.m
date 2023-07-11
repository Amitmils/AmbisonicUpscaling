%% This script analyzes BSM performance with random signals (numerical errors)

% Date created: May 2021
% Created by:   Lior Madmoni

clearvars;
close all;
clc;
restoredefaultpath;

% add ACLtoolbox path
addpath(genpath('/Users/liormadmoni/Google Drive/Lior/Acoustics lab/Matlab/Research/Github/general'));
cd('/Users/liormadmoni/Google Drive/Lior/Acoustics lab/Matlab/Research/Github/general/');

% add export_fig to path
addpath(genpath('/Volumes/GoogleDrive/My Drive/Lior/Acoustics lab/Matlab/Research/FB_BFBR/Toolboxes/altmany-export_fig-9aba302'));

% add sparse recovery scripts and CVX toolbox
addpath(genpath('/Volumes/GoogleDrive/My Drive/Lior/Acoustics lab/Matlab/Research/FB_BFBR/Sparse_recovery/l0_approximation/'));
addpath(genpath('/Volumes/GoogleDrive/My Drive/Lior/Acoustics lab/Matlab/Research/Solvers/CVX/'));

startup_script();
rng('default');

%% ================== simulation parameters
% parameters/flags - array
filt_len = 0.032;                                      % filters (BSM/HRTF) length [sec]
arrayType = 1;                                         % 0 - spherical array, 1 - semi-circular array, 2 - full-circular array
rigidArray = 1;                                        % 0 - open array, 1 - rigid array
M = 6;                                                 % number of microphones
r_array = 0.1;                                         % array radius
head_rot_az = ...
    wrapTo2Pi(deg2rad([0]));                         % vector of head rotations [rad]
normSV = true;                                         % true - normalize steering vectors

% parameters/flags - general
c = 343;                                               % speed of sound [m/s]
N_PW = 14;                                             % SH order of plane-wave synthesis

% parameters/flags - BSM design
BSM_inv_opt = 2;                                       % 1 - ( (1 / lambda) * (A * A') + eye ),  2 - ((A * A') + lambda * eye);
source_distribution = 1;                               % 0 - nearly uniform (t-design), 1 - spiral nearly uniform
Q = 240;                                               % Assumed number of sources

% parameters/flags - noise (regularization)
SNR = 50;                                              % assumed sensors SNR [dB]
sigma_s = 1;
sigma_n = 10^(-SNR/20) * sigma_s;
SNR_lin = (sigma_s / sigma_n)^2;    

% Sparse recovery parameters
omp_sigma = 1e-6;

irls_lambda = 1e-1;     % a sweep required to choose the correct value
irls_delta = 1e-6;      % regularization of inverse matrix
irls_thr = 1e-4;        % convergence between consecutive solutions
irls_print_iter = false;

l1_eps = 1 / SNR_lin;

% Text variables for plots 
if ~rigidArray
    sphereType = 'open';
else
    sphereType = 'rigid';
end
if arrayType == 0
    arrTypeTxt = 'spherical';
elseif arrayType == 1
    arrTypeTxt = 'semi-cir';
elseif arrayType == 2
    arrTypeTxt = 'fully-circ';
end

% BSM grid
[th_BSMgrid_vec, ph_BSMgrid_vec] = BSM_toolbox.BSMgrid(source_distribution, Q);

% BSM filter length and frequencies
desired_fs = 16000;
filt_samp = filt_len * desired_fs;
freqs_sig = ( 0 : (filt_samp / 2) ) * desired_fs / filt_samp;
freqs_sig(1) = 1 / 4 * freqs_sig(2);

%% ================== Create BSM struct
BSMobj.freqs_sig = freqs_sig;
BSMobj.N_PW = N_PW;    
BSMobj.c = c;
BSMobj.r_array = r_array;
BSMobj.rigidArray = rigidArray;
BSMobj.th_BSMgrid_vec = th_BSMgrid_vec;
BSMobj.ph_BSMgrid_vec = ph_BSMgrid_vec;
%
BSMobj.normSV = normSV;
BSMobj.SNR_lin = SNR_lin;
BSMobj.inv_opt = BSM_inv_opt;
BSMobj.head_rot_az = head_rot_az;
BSMobj.M = M;
BSMobj.Q = Q;
BSMobj.source_distribution = source_distribution;
BSMobj.desired_fs = desired_fs;
BSMobj.filt_samp = filt_samp;
BSMobj.sphereType = sphereType;

%% ================= Get array positions
n_mic = M;        
[th_array, ph_array, ~] = BSM_toolbox.GetArrayPositions(arrayType, n_mic, 0);

%% ================== Update BSM struct
BSMobj.n_mic = n_mic;
BSMobj.th_array = th_array;
BSMobj.ph_array = ph_array;      

%% ================== BSM directions
% SH order of steering-vectors
N_SV = N_PW;
[th_BSMgrid_vec, ph_BSMgrid_vec] = BSM_toolbox.BSMgrid(source_distribution, Q);
V_BSM = CalculateSteeringVectors(BSMobj, N_SV, th_BSMgrid_vec, ph_BSMgrid_vec);
V_BSM = permute(V_BSM, [3 2 1]);

%% ================== load HRIRs
N_HRTF = 30;
HRTFpath =  '/Users/liormadmoni/Google Drive/ACLtoolbox/Data/HRTF/earoHRIR_KU100_Measured_2702Lebedev.mat';
load(HRTFpath);         % hobj is HRIR earo object
hobj.shutUp = true;
hobj_full = hobj;

% Transform to frequency domain (HRTFs)
if strcmp(hobj_full.dataDomain{1},'FREQ'), hobj_full = hobj_full.toTime(); end

% Resample HRTF to desired_fs
hobj_full = hobj_full.resampleData(desired_fs);
hobj_full = hobj_full.toFreq(filt_samp);

% Trim negative frequencies
hobj_full.data = hobj_full.data(:, 1:ceil(filt_samp/2)+1, :);

% Rotate HRTFs
WignerDpath = '/Volumes/GoogleDrive/My Drive/ACLtoolbox/Data/WignerDMatrix_diagN=32.mat';
load(WignerDpath);
N_HRTF_rot = N_HRTF;
DN = (N_HRTF_rot + 1)^2; % size of the wignerD matrix
D_allAngles = D(:, 1 : DN);
hobj_full_rot = RotateHRTF(hobj_full, N_HRTF_rot, D_allAngles, head_rot_az);
hobj_full_rot = hobj_full_rot.toSpace('SRC');

%% ================= source and binaural signals estimation - known source directions
%  ================= head rotation by HRTF rotation
%{
rng('default');
T = 100; % snapshots
experiments = 10;
L = 6;
add_noise = false;
rotate_head = true;
save_plot = false;

err_s = zeros(experiments, length(freqs_sig));
err_p = zeros(experiments, length(freqs_sig), 2);
var_p = zeros(experiments, length(freqs_sig), 2);
var_p_hat = zeros(experiments, length(freqs_sig), 2);
for e = 1:experiments
    % Generate random DOAs
    th_s = rand(L, 1) * pi;
    ph_s = rand(L, 1) * 2 * pi;
    
    % Calculate steering vectors
    V_k = CalculateSteeringVectors(BSMobj, N_SV, th_s,  ph_s);
    V_k = permute(V_k, [3 2 1]);
    
    % calculate HRTFs
    if ~rotate_head
        hobj_true_sources = BSM_toolbox.HRTF2BSMgrid(hobj_full, N_HRTF, th_s, ph_s);
    else
        hobj_true_sources_rot = BSM_toolbox.HRTF2BSMgrid(hobj_full_rot, N_HRTF, th_s, ph_s);
    end

    for f = 1:length(freqs_sig)
        % Random signals
        s = sigma_s * randn(L, T);
        Rs = cov(s.');

        % Noise
        n = sigma_n * randn(M, T);
        Rn = cov(n.');        
        
        % Measured signals
        V = V_k(:, : , f);
        V = V ./ vecnorm(V);
        if add_noise
            x_f = V * s + n;
        else
            x_f = V * s;
        end

        % Signal estimation
%         s_hat = Rs * V' * pinv(V * Rs * V' + Rn) * x_f;
        s_hat = V' * pinv(V * V' + 1 / SNR_lin * eye(M)) * x_f;
        err_s(e, f) = sqrt(mean(vecnorm(s - s_hat).^2)) ./ ...
            sqrt(mean(vecnorm(s).^2));
        
        % HRTFs
        if ~rotate_head
            h_true = squeeze(hobj_true_sources.data(:, f, :));
        else
            h_true = squeeze(hobj_true_sources_rot.data(:, f, :));
        end  
        
        % Binaural signals estimation
        p = h_true.' * s;
        p_hat = h_true.' * s_hat;
        err_p(e, f, :) = vecnorm(p.' - p_hat.') ./ vecnorm(p.');
        var_p(e, f, :) = var(p.');
        var_p_hat(e, f, :) = var(p_hat.');
    end
end
err_s = sqrt(mean(err_s.^2, 1));
err_p = squeeze(sqrt(mean(err_p.^2, 1)));
var_p = squeeze(mean(var_p, 1));
var_p_hat = squeeze(mean(var_p_hat, 1));

% error plot
figure;
% semilogx(freqs_sig, mag2db(err_s), 'linewidth', 2);
hold on;
semilogx(freqs_sig, mag2db(err_p), 'linewidth', 3);
xlabel('Frequency [Hz]');
ylabel('Error [dB]');
title(['TR-BSM known directions, ', num2str(L), ' sources, $\phi_{rot}$=',...
    num2str(rad2deg(head_rot_az)), '$^{\circ}$']);
% legend({'$\epsilon_s$','$\epsilon_p^l$','$\epsilon_p^r$'});
legend({'$\epsilon_p^l$','$\epsilon_p^r$'});
if save_plot
%     export_fig(['/Users/liormadmoni/Google Drive/Lior/Acoustics lab/Matlab/',...
%         'Research/FB_BFBR/BSM/plots/numerical_errors/errors/',...
%         'known_sources_semicircM=6_L=',num2str(L),...
%         '_rot=',num2str(rad2deg(head_rot_az)),'.png'],...
%         '-transparent', '-r300');

    export_fig(['/Volumes/GoogleDrive/My Drive/Lior/Acoustics lab/Research/',...
            'FB/Binaural_beamforming/Presentations/pres12/figs/',...
            'err_known_sources_semicircM=6_L=',num2str(L),...
            '_rot=',num2str(rad2deg(head_rot_az)),'.png'],...
            '-transparent', '-r300');
end

% var plot
figure;
semilogx(freqs_sig, var_p(:, 1), 'color', '#0072BD');
hold on;
semilogx(freqs_sig, var_p(:, 2), 'color', '#D95319');
semilogx(freqs_sig, var_p_hat(:, 1), '-+', 'color', '#0072BD');
semilogx(freqs_sig, var_p_hat(:, 2), '-+', 'color', '#D95319');
xlabel('Frequency [Hz]');
ylabel('Variance');
title(['TR-BSM known directions, ', num2str(L), ' sources, $\phi_{rot}$=',...
    num2str(rad2deg(head_rot_az)), '$^{\circ}$']);
legend({'var[$p^l$]', 'var[$p^r$]', 'var[$\hat{p}^l$]', 'var[$\hat{p}^r$]'});
if save_plot
%     export_fig(['/Users/liormadmoni/Google Drive/Lior/Acoustics lab/Matlab/',...
%         'Research/FB_BFBR/BSM/plots/numerical_errors/vars/',...
%         'known_sources_semicircM=6_L=',num2str(L),...
%         '_rot=',num2str(rad2deg(head_rot_az)),'.png'],...
%         '-transparent', '-r300');
    
    export_fig(['/Volumes/GoogleDrive/My Drive/Lior/Acoustics lab/Research/',...
            'FB/Binaural_beamforming/Presentations/pres12/figs/',...
            'var_known_sources_semicircM=6_L=',num2str(L),...
            '_rot=',num2str(rad2deg(head_rot_az)),'.png'],...
            '-transparent', '-r300');
end
%}

%% ================= binaural signals estimation - using BSM directions
%  ================= head rotation by HRTF rotation
%{
rng('default');
T = 100; % snapshots
experiments = 10;
L = 6;
add_noise = false;
rotate_head = true;
save_plot = false;

% BSM directions
source_distribution = 1;
Q = 240;
[th_BSMgrid_vec, ph_BSMgrid_vec] = BSM_toolbox.BSMgrid(source_distribution, Q);
V_BSM = CalculateSteeringVectors(BSMobj, N_SV, th_BSMgrid_vec, ph_BSMgrid_vec);
V_BSM = permute(V_BSM, [3 2 1]);

err_p = zeros(experiments, length(freqs_sig), 2);
var_p = zeros(experiments, length(freqs_sig), 2);
var_p_hat = zeros(experiments, length(freqs_sig), 2);
c_BSM_energy = zeros(experiments, length(freqs_sig));
for e = 1:experiments
    % generate random DOAs
    th_s = rand(L, 1) * pi;
    ph_s = rand(L, 1) * 2 * pi;
    
    % calculate steering vectors
    V_k = CalculateSteeringVectors(BSMobj, N_SV, th_s,  ph_s);
    V_k = permute(V_k, [3 2 1]);
    
    % calculate HRTFs
    if ~rotate_head
        hobj_true_sources = BSM_toolbox.HRTF2BSMgrid(hobj_full, N_HRTF, th_s, ph_s);
        hobj_bsm_directions = BSM_toolbox.HRTF2BSMgrid(hobj_full, N_HRTF, th_BSMgrid_vec, ph_BSMgrid_vec);
    else
        hobj_true_sources_rot = BSM_toolbox.HRTF2BSMgrid(hobj_full_rot, N_HRTF, th_s, ph_s);
        hobj_bsm_directions_rot = BSM_toolbox.HRTF2BSMgrid(hobj_full_rot, N_HRTF, th_BSMgrid_vec, ph_BSMgrid_vec);
    end

    for f = 1:length(freqs_sig)
        % random signals
        s = sigma_s * randn(L, T);
        Rs = cov(s.');

        % noise
        n = sigma_n * randn(M, T);
        Rn = cov(n.');        
        
        V = V_k(:, : , f);
        V = V ./ vecnorm(V);
        if add_noise
            x_f = V * s + n;
        else
            x_f = V * s;
        end
        
        % use BSM directions
        V = V_BSM(:, :, f);
        V = V ./ vecnorm(V);
        
        % signal estimation
        s_hat = V' * pinv(V * V' + 1 / SNR_lin * eye(M)) * x_f;
        
        % HRTFs
        if ~rotate_head
            h_true = squeeze(hobj_true_sources.data(:, f, :));
            h_bsm = squeeze(hobj_bsm_directions.data(:, f, :));
        else
            h_true = squeeze(hobj_true_sources_rot.data(:, f, :));
            h_bsm = squeeze(hobj_bsm_directions_rot.data(:, f, :));
        end        
        
        % binaural signals estimation
        p = h_true.' * s;
        p_hat = h_bsm.' * s_hat;
        err_p(e, f, :) = vecnorm(p.' - p_hat.') ./ vecnorm(p.');
        var_p(e, f, :) = var(p.');
        var_p_hat(e, f, :) = var(p_hat.');
    end
end
err_p = squeeze(sqrt(mean(err_p.^2, 1)));
var_p = squeeze(mean(var_p, 1));
var_p_hat = squeeze(mean(var_p_hat, 1));

% error plot
figure;
semilogx(freqs_sig, mag2db(err_p), 'linewidth', 3);
hold on;
% ylim([-30 1]);
xlabel('Frequency [Hz]');
ylabel('Error [dB]');
title(['TR-BSM, ', num2str(L), ' sources, $\phi_{rot}$=',...
    num2str(rad2deg(head_rot_az)), '$^{\circ}$']);
legend({'$\epsilon_p^l$', '$\epsilon_p^r$'}, 'interpreter', 'latex');
if save_plot
%     export_fig(['/Users/liormadmoni/Google Drive/Lior/Acoustics lab/Matlab/',...
%         'Research/FB_BFBR/BSM/plots/numerical_errors/errors/',...
%         'unknown_sources_semicircM=6_L=',num2str(L),...
%         '_rot=',num2str(rad2deg(head_rot_az)),'.png'],...
%         '-transparent', '-r300');

    export_fig(['/Volumes/GoogleDrive/My Drive/Lior/Acoustics lab/Research/',...
            'FB/Binaural_beamforming/Presentations/pres12/figs/',...
            'err_unknown_sources_semicircM=6_L=',num2str(L),...
            '_rot=',num2str(rad2deg(head_rot_az)),'.png'],...
            '-transparent', '-r300');
end

% var plot
%{
figure;
semilogx(freqs_sig, var_p(:, 1), 'color', '#0072BD');
hold on;
semilogx(freqs_sig, var_p(:, 2), 'color', '#D95319');
semilogx(freqs_sig, var_p_hat(:, 1), '-+', 'color', '#0072BD');
semilogx(freqs_sig, var_p_hat(:, 2), '-+', 'color', '#D95319');
xlabel('Frequency [Hz]');
ylabel('Variance');
title(['TR-BSM, ', num2str(L), ' sources, $\phi_{rot}$=',...
    num2str(rad2deg(head_rot_az)), '$^{\circ}$']);
legend({'var[$p^l$]', 'var[$p^r$]', 'var[$\hat{p}^l$]', 'var[$\hat{p}^r$]'});
if save_plot
%     export_fig(['/Users/liormadmoni/Google Drive/Lior/Acoustics lab/Matlab/',...
%         'Research/FB_BFBR/BSM/plots/numerical_errors/vars/',...
%         'unknown_sources_semicircM=6_L=',num2str(L),...
%         '_rot=',num2str(rad2deg(head_rot_az)),'.png'],...
%         '-transparent', '-r300');

    export_fig(['/Volumes/GoogleDrive/My Drive/Lior/Acoustics lab/Research/',...
            'FB/Binaural_beamforming/Presentations/pres12/figs/',...
            'var_unknown_sources_semicircM=6_L=',num2str(L),...
            '_rot=',num2str(rad2deg(head_rot_az)),'.png'],...
            '-transparent', '-r300');
end
%}
%}

%% ================= binaural signals estimation - using BSM directions
%  ================= head rotation by steering matrix rotation
%{
rng('default');
T = 100; % snapshots
experiments = 10;
L = 7;
add_noise = false;
rotate_head = true;
save_plot = false;

if rotate_head
    V_BSM_rot = CalculateSteeringVectors(BSMobj, N_SV, th_BSMgrid_vec, wrapTo2Pi(ph_BSMgrid_vec + head_rot_az));
    V_BSM_rot = permute(V_BSM_rot, [3 2 1]);
end

err_p = zeros(experiments, length(freqs_sig), 2);
for e = 1:experiments
    % generate random DOAs
    th_s = rand(L, 1) * pi;
    ph_s = rand(L, 1) * 2 * pi;
    
    % calculate steering vectors
    V_k = CalculateSteeringVectors(BSMobj, N_SV, th_s,  ph_s);
    V_k = permute(V_k, [3 2 1]);
    
    % calculate HRTFs
    if ~rotate_head
        hobj_true_sources = BSM_toolbox.HRTF2BSMgrid(hobj_full, N_HRTF, th_s, ph_s);    
    else
        hobj_true_sources_rot = BSM_toolbox.HRTF2BSMgrid(hobj_full_rot, N_HRTF, th_s, ph_s);
    end
    hobj_bsm_directions = BSM_toolbox.HRTF2BSMgrid(hobj_full, N_HRTF, th_BSMgrid_vec, ph_BSMgrid_vec);
    
    for f = 1:length(freqs_sig)
        % random signals
        s = sigma_s * randn(L, T);
        Rs = cov(s.');

        % noise
        n = sigma_n * randn(M, T);
        Rn = cov(n.');        
        
        V = V_k(:, : , f);
        V = V ./ vecnorm(V);
        if add_noise
            x_f = V * s + n;
        else
            x_f = V * s;
        end
        
        % use BSM directions
        if ~rotate_head
            V = V_BSM(:, :, f);            
        else
            V = V_BSM_rot(:, :, f);
        end
        V = V ./ vecnorm(V);
        
        % signal estimation
        s_hat = V' * pinv(V * V' + 1 / SNR_lin * eye(M)) * x_f;
        
        % HRTFs
        if ~rotate_head
            h_true = squeeze(hobj_true_sources.data(:, f, :));            
        else
            h_true = squeeze(hobj_true_sources_rot.data(:, f, :));
        end
        h_bsm = squeeze(hobj_bsm_directions.data(:, f, :));
        
        % binaural signals estimation
        p = h_true.' * s;
        p_hat = h_bsm.' * s_hat;
        err_p(e, f, :) = vecnorm(p.' - p_hat.') ./ vecnorm(p.');
    end
end
err_p = squeeze(sqrt(mean(err_p.^2, 1)));

% error plot
figure;
semilogx(freqs_sig, mag2db(err_p), 'linewidth', 3);
hold on;
ylim([-30 1]);
xlabel('Frequency [Hz]');
ylabel('Error [dB]');
title(['TR-BSM, ', num2str(L), ' sources, SNR=', num2str(SNR), ' dB']);
legend({'$\epsilon_p^l$', '$\epsilon_p^r$'}, 'interpreter', 'latex');
if save_plot
    export_fig(['/Users/liormadmoni/Google Drive/Lior/Acoustics lab/Matlab/',...
        'Research/FB_BFBR/BSM/plots/numerical_errors/',...
        'unknown_sources_semicircM=6_L=',num2str(L),...
        '_rot=',num2str(rad2deg(head_rot_az)),'_steering.png'],...
        '-transparent', '-r300');
end
%}

%% ================= binaural signals estimation - using BSM directions and sparse recovery
%  ================= head rotation by HRTF rotation
%{
rng('default');
T = 100; % snapshots
experiments = 10;
L = 6;
add_noise = false;
rotate_head = true;
sparse_method = 'L1';  % either OMP, IRLS, L1

if strcmp(sparse_method, 'OMP')
    omp_sigma = 1 / SNR_lin;
elseif strcmp(sparse_method, 'IRLS')
    irls_delta = 1e-9;
    irls_thr = 1 / SNR_lin;
    irls_lambda = 0.01;
    irls_print_iter = false;
elseif strcmp(sparse_method, 'L1')
    l1_eps = 1 / SNR_lin;
else
    disp('Sparse-recovery method error: choose OMP/IRLS/L1. Using TR instead!')
end

err_p = zeros(experiments, length(freqs_sig), 2);
for e = 1:experiments
    % generate random DOAs
    th_s = rand(L, 1) * pi;
    ph_s = rand(L, 1) * 2 * pi;
    
    % calculate steering vectors
    V_k = CalculateSteeringVectors(BSMobj, N_SV, th_s,  ph_s);
    V_k = permute(V_k, [3 2 1]);
    
    % calculate HRTFs
    if ~rotate_head
        hobj_true_sources = BSM_toolbox.HRTF2BSMgrid(hobj_full, N_HRTF, th_s, ph_s);
        hobj_bsm_directions = BSM_toolbox.HRTF2BSMgrid(hobj_full, N_HRTF, th_BSMgrid_vec, ph_BSMgrid_vec);
    else
        hobj_true_sources_rot = BSM_toolbox.HRTF2BSMgrid(hobj_full_rot, N_HRTF, th_s, ph_s);
        hobj_bsm_directions_rot = BSM_toolbox.HRTF2BSMgrid(hobj_full_rot, N_HRTF, th_BSMgrid_vec, ph_BSMgrid_vec);
    end

    for f = 1:length(freqs_sig)
        % random signals
        s = sigma_s * randn(L, T);
        Rs = cov(s.');

        % noise
        n = sigma_n * randn(M, T);
        Rn = cov(n.');        
        
        V = V_k(:, : , f);
        V = V ./ vecnorm(V);
        if add_noise
            x_f = V * s + n;
        else
            x_f = V * s;
        end
        
        % use BSM directions
        V = V_BSM(:, :, f);
        V = V ./ vecnorm(V);
        
        % signal estimation with sparse method
        if strcmp(sparse_method, 'OMP')
            s_hat = zeros(Q, T);
            for t=1:T
                [xOMP, choice, Sopt] = OMP(V, x_f(:, t), omp_sigma);
                if ~isempty(choice)
                    s_hat(:, t) = xOMP(:, choice);
                else
                    s_hat(:, t) = zeros(Q, 1);
                end
            end
        elseif strcmp(sparse_method, 'IRLS')
            s_hat = zeros(Q, T);
            for t=1:T
                s_hat(:, t) = IRLS_for_basisPursuit(V, x_f(:, t), ...
                    irls_lambda, irls_delta, irls_thr, irls_print_iter);
            end
        elseif strcmp(sparse_method, 'L1')
            s_hat = zeros(Q, T);
            for t=1:T
                cvx_begin quiet
                variable alpha_hat(Q, 1)
                V * alpha_hat == x_f(:, t)
                minimize(norm(alpha_hat, 1))                
                cvx_end
                
                s_hat(:, t) = alpha_hat;
            end
        else
            % signal estimation with Tikhonov regularization
            s_hat = V' * pinv(V * V' + 1 / SNR_lin * eye(M)) * x_f;
        end
            
        % HRTFs
        if ~rotate_head
            h_true = squeeze(hobj_true_sources.data(:, f, :));
            h_bsm = squeeze(hobj_bsm_directions.data(:, f, :));
        else
            h_true = squeeze(hobj_true_sources_rot.data(:, f, :));
            h_bsm = squeeze(hobj_bsm_directions_rot.data(:, f, :));
        end        
        
        % binaural signals estimation
        p = h_true.' * s;
        p_hat = h_bsm.' * s_hat;
        err_p(e, f, :) = vecnorm(p.' - p_hat.') ./ vecnorm(p.');
    end
end
err_p = squeeze(sqrt(mean(err_p.^2, 1)));
% s_hat(abs(s_hat).^2 < 1 / SNR_lin) = 0;

% error plot
figure;
semilogx(freqs_sig, mag2db(err_p), 'linewidth', 3);
hold on;
ylim([-30 1]);
xlabel('Frequency [Hz]');
ylabel('Error [dB]');
title([sparse_method, '-BSM, ', num2str(L), ' sources, SNR=', num2str(SNR), ' dB']);
legend({'$\epsilon_p^l$', '$\epsilon_p^r$'}, 'interpreter', 'latex');

%}

%% ================= BSM vs sparse (OMP, IRLS, L1)
%  ================= head rotation by HRTF rotation
%
rng('default');
T = 2; % snapshots
experiments = 1;
add_noise = false;
rotate_head = false;
save_plot = false;

% BSM directions
source_distribution = 1;
Q = 1000;
L = 10;
[th_BSMgrid_vec, ph_BSMgrid_vec] = BSM_toolbox.BSMgrid(source_distribution, Q);
V_BSM = CalculateSteeringVectors(BSMobj, N_SV, th_BSMgrid_vec, ph_BSMgrid_vec);
V_BSM = permute(V_BSM, [3 2 1]);

% override to look at subset of frequencies
%{
freqs_sig = ( 0 : (filt_samp / 2) ) * desired_fs / filt_samp;
freqs_sig(1) = 1 / 4 * freqs_sig(2);
freqs_sig = freqs_sig([4, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250]);
%}

% sparse parameters
sparse_method = 'L1';

err_p_BSM = zeros(length(freqs_sig), 2, experiments);
err_p_sparse = zeros(length(freqs_sig), 2, experiments);
err_p_norm_BSM = zeros(length(freqs_sig), 2, experiments);
err_p_norm_sparse = zeros(length(freqs_sig), 2, experiments);

err_s_BSM = zeros(length(freqs_sig), experiments);
err_s_sparse = zeros(length(freqs_sig), experiments);
err_s_norm_BSM = zeros(length(freqs_sig), experiments);
err_s_norm_sparse = zeros(length(freqs_sig), experiments);

for e = 1:experiments
    % init
    p = zeros(length(freqs_sig), 2, T);
    p_hat_sparse = zeros(length(freqs_sig), 2, T);
    p_hat_BSM = zeros(length(freqs_sig), 2, T);
    p_norm = zeros(length(freqs_sig), 2, T);
    p_hat_norm_sparse = zeros(length(freqs_sig), 2, T);
    p_hat_norm_BSM = zeros(length(freqs_sig), 2, T);

    s_hat_sparse = zeros(length(freqs_sig), Q, T);
    s_hat_BSM = zeros(length(freqs_sig), Q, T);
    s_hat_norm_sparse = zeros(length(freqs_sig), Q, T);
    s_hat_norm_BSM = zeros(length(freqs_sig), Q, T);
      
    % choose L source directions
    active = round(rand(L, 1) * length(th_BSMgrid_vec));
    %active = [120, 121, 122, 123].';
    %active = 500:(500+L - 1); active = active.';
    s_e = zeros(length(th_BSMgrid_vec), T);
    s_e(active, :) = (randn(length(active), T) + 1j * randn(length(active), T));
    s_e = sigma_s * s_e ./ vecnorm(s_e);
     
    % calculate HRTFs
    if ~rotate_head
        hobj_bsm_directions = BSM_toolbox.HRTF2BSMgrid(hobj_full, N_HRTF, th_BSMgrid_vec, ph_BSMgrid_vec);
    else
        hobj_bsm_directions_rot = BSM_toolbox.HRTF2BSMgrid(hobj_full_rot, N_HRTF, th_BSMgrid_vec, ph_BSMgrid_vec);
    end

    for f = 1:length(freqs_sig)   
        %% OVERRIDE TO ANALYZE SINGAL FREQUENCY
        %{
        f_analyze = 2000;
        [~, f_ind_analyze] = min(abs(freqs_sig - f_analyze));
        f = f_ind_analyze;
        %}
        
        % covariance matrix of sources
        %Rs = cov(s.'); 

        % use BSM directions
        V = V_BSM(:, :, f);
        %V = V ./ vecnorm(V);
        if add_noise
            n = sigma_n * randn(M, T);            
            x_f = V * s_e + n;
            %Rn = cov(n.');
        else
            x_f = V * s_e;
        end           
        
        % signal estimation with BSM
        s_hat_BSM_e = V' * pinv(V * V' + 1 / SNR_lin * eye(M)) * x_f;
        
        % signal estimation with sparse methods
        s_hat_sparse_e = zeros(Q, T);
        if strcmp(sparse_method, 'OMP')
            % OMP
            for t=1:T
                pf = squeeze(x_f(:, t));
                pf_norm = vecnorm(pf);
                if pf_norm
                    pf = pf / pf_norm;
                end

                [xOMP, choice, ~, ~] = OMP(V, pf, omp_sigma);
                s_hat_OMP = xOMP(:, choice);
                if ~isempty(s_hat_OMP)
                    s_hat_sparse_e(:, t) = pf_norm * s_hat_OMP;   
                    %s_hat_sparse_e(:, t) = s_hat_OMP;
                    %s_hat_sparse_e(:, t) = pf_norm / vecnorm(s_hat_sparse_e(:, t)) * s_hat_sparse_e(:, t);
                end
            end
        elseif strcmp(sparse_method, 'IRLS')
            % IRLS
            for t=1:T
                pf = squeeze(x_f(:, t));
                pf_norm = vecnorm(pf);
                if pf_norm
                    pf = pf / pf_norm;
                end

                s_hat_IRLS = IRLS_for_basisPursuit(V, pf, ...
                    irls_lambda, irls_delta, irls_thr, irls_print_iter);                
                s_hat_sparse_e(:, t) = pf_norm * s_hat_IRLS;
                %s_hat_sparse_e(:, t) = s_hat_IRLS;
                %s_hat_sparse_e(:, t) = pf_norm / vecnorm(s_hat_sparse_e(:, t)) * s_hat_sparse_e(:, t);
            end
        elseif strcmp(sparse_method, 'L1')
            % L1
            for t=1:T
                pf = squeeze(x_f(:, t));
                pf_norm = vecnorm(pf);
                if pf_norm
                    pf = pf / pf_norm;
                end

                % CVX solver
                cvx_begin quiet
                    variable alpha_hat(Q, 1) complex
                    V * alpha_hat == pf
                    minimize(norm(alpha_hat, 1))
                cvx_end
                
                s_hat_L1 = alpha_hat;                
                s_hat_sparse_e(:, t) = pf_norm * s_hat_L1;
                %s_hat_sparse_e(:, t) = s_hat_L1;
                %s_hat_sparse_e(:, t) = pf_norm / vecnorm(s_hat_sparse_e(:, t)) * s_hat_sparse_e(:, t);
            end
        else
            % signal estimation with Tikhonov regularization
            fprintf('Sparse recovery method has to be one of the following: OMP/L1/IRLS!\n');
        end
        
        % HRTFs
        if ~rotate_head
            h_bsm = squeeze(hobj_bsm_directions.data(:, f, :));
        else
            h_bsm = squeeze(hobj_bsm_directions_rot.data(:, f, :));
        end
        
        % normalize signals
        s_e_norm = s_e ./ vecnorm(s_e, 2, 1);
        s_hat_BSM_e_norm = s_hat_BSM_e ./ vecnorm(s_hat_BSM_e, 2, 1);
        s_hat_sparse_e_norm = s_hat_sparse_e ./ vecnorm(s_hat_sparse_e, 2, 1);
        
        % sound source estimation
        s_hat_BSM(f, :, :) = s_hat_BSM_e; 
        s_hat_sparse(f, :, :) = s_hat_sparse_e;
        s_hat_norm_BSM(f, :, :) = s_hat_BSM_e_norm;
        s_hat_norm_sparse(f, :, :) = s_hat_sparse_e_norm;
        
        % binaural signals estimation
        p(f, :, :) = h_bsm.' * s_e;
        p_hat_BSM(f, :, :) = h_bsm.' * s_hat_BSM_e;
        p_hat_sparse(f, :, :) = h_bsm.' * s_hat_sparse_e;
        
        p_norm(f, :, :) = h_bsm.' * s_e_norm;
        p_hat_norm_BSM(f, :, :) = h_bsm.' * s_hat_BSM_e_norm;
        p_hat_norm_sparse(f, :, :) = h_bsm.' * s_hat_sparse_e_norm;
        
        % print status
        if f == 1
            fprintf('finished %s calculation for exp %d/%d frequency index %d/%d', sparse_method, e, experiments, f, length(freqs_sig));
        else
            fprintf(repmat('\b', 1, (length(num2str(length(freqs_sig))) + 1 + (length(num2str(f - 1))))));
            fprintf('%d/%d',f, length(freqs_sig));
        end
    end
    fprintf('\n');
    
    % performance metrics
    err_p_BSM(:, :, e) = vecnorm(p - p_hat_BSM, 2, 3) ./ vecnorm(p, 2, 3);
    err_p_sparse(:, :, e) = vecnorm(p - p_hat_sparse, 2, 3) ./ vecnorm(p, 2, 3);
    err_p_norm_BSM(:, :, e) = vecnorm(p_norm - p_hat_norm_BSM, 2, 3) ./ vecnorm(p_norm, 2, 3);
    err_p_norm_sparse(:, :, e) = vecnorm(p_norm - p_hat_norm_sparse, 2, 3) ./ vecnorm(p_norm, 2, 3);

    for f=1:length(freqs_sig)
        err_s_BSM(f, e) = vecnorm(vecnorm(s_e - squeeze(s_hat_BSM(f, :, :)), 2, 2));
        err_s_sparse(f, e) = vecnorm(vecnorm(s_e - squeeze(s_hat_sparse(f, :, :)), 2, 2));
        err_s_norm_BSM(f, e) = vecnorm(vecnorm(s_e_norm - squeeze(s_hat_norm_BSM(f, :, :)), 2, 2));
        err_s_norm_sparse(f, e) = vecnorm(vecnorm(s_e_norm - squeeze(s_hat_norm_sparse(f, :, :)), 2, 2));
    end    
end

% average error on all experiments
if experiments > 1
    err_p_BSM = vecnorm(err_p_BSM, 2, 3);
    err_p_sparse = vecnorm(err_p_sparse, 2, 3);
    err_p_norm_BSM = vecnorm(err_p_norm_BSM, 2, 3);
    err_p_norm_sparse = vecnorm(err_p_norm_sparse, 2, 3);
    
    err_s_BSM = vecnorm(err_s_BSM, 2, 2);
    err_s_sparse = vecnorm(err_s_sparse, 2, 2);
    err_s_norm_BSM = vecnorm(err_s_norm_BSM, 2, 2);
    err_s_norm_sparse = vecnorm(err_s_norm_sparse, 2, 2);
end

%%save_plot = true;
%%Binaural signal error plot
figure;
c_order = num2cell(get(gca,'colororder'), 2);
bbl = semilogx(freqs_sig, mag2db(err_p_BSM(:, 1)), 'linewidth', 3, 'linestyle', '-', 'DisplayName', 'BSM-left');
hold on;
bbr = semilogx(freqs_sig, mag2db(err_p_BSM(:, 2)), 'linewidth', 3, 'linestyle', ':', 'DisplayName', 'BSM-right');
bsl = semilogx(freqs_sig, mag2db(err_p_sparse(:, 1)), 'linewidth', 3, 'linestyle', '-', 'DisplayName', [sparse_method,'-left']);
bsr = semilogx(freqs_sig, mag2db(err_p_sparse(:, 2)), 'linewidth', 3, 'linestyle', ':', 'DisplayName', [sparse_method,'-right']);
set(bbl, {'color'}, c_order(1, :));
set(bbr, {'color'}, c_order(1, :));
set(bsl, {'color'}, c_order(2, :));
set(bsr, {'color'}, c_order(2, :));
% ylim([-60 5]);
axis tight;
xlabel('Frequency [Hz]');
ylabel('Binaural Signal Error [dB]');
title([num2str(L), ' sources, $\phi_{rot}$=',...
    num2str(rad2deg(head_rot_az)), '$^{\circ}$']);
legend('location', 'northwest', 'interpreter', 'latex');
% legend({'BSM-l', 'BSM-r', [sparse_method,'-l'], [sparse_method,'-r']}, 'interpreter', 'latex');
if save_plot
    export_fig(['/Volumes/GoogleDrive/My Drive/Lior/Acoustics lab/Matlab/',...
        'Research/FB_BFBR/BSM/plots/numerical_errors/sparse/',...
        'binaural_err_L=',num2str(L),'_',arrTypeTxt,'.png'], '-transparent', '-r300');
end

%%Normalized Binaural signal error plot
%{
figure;
c_order = num2cell(get(gca,'colororder'), 2);
bbl = semilogx(freqs_sig, mag2db(err_p_norm_BSM(:, 1)), 'linewidth', 3, 'linestyle', '-', 'DisplayName', 'BSM-left');
hold on;
bbr = semilogx(freqs_sig, mag2db(err_p_norm_BSM(:, 2)), 'linewidth', 3, 'linestyle', ':', 'DisplayName', 'BSM-right');
bsl = semilogx(freqs_sig, mag2db(err_p_norm_sparse(:, 1)), 'linewidth', 3, 'linestyle', '-', 'DisplayName', [sparse_method,'-left']);
bsr = semilogx(freqs_sig, mag2db(err_p_norm_sparse(:, 2)), 'linewidth', 3, 'linestyle', ':', 'DisplayName', [sparse_method,'-right']);
set(bbl, {'color'}, c_order(1, :));
set(bbr, {'color'}, c_order(1, :));
set(bsl, {'color'}, c_order(2, :));
set(bsr, {'color'}, c_order(2, :));
% ylim([-60 5]);
axis tight;
xlabel('Frequency [Hz]');
ylabel('Normalized Binaural Signal Error [dB]');
title([num2str(L), ' sources, $\phi_{rot}$=',...
    num2str(rad2deg(head_rot_az)), '$^{\circ}$']);
legend('location', 'northwest', 'interpreter', 'latex');
% legend({'BSM-l', 'BSM-r', [sparse_method,'-l'], [sparse_method,'-r']}, 'interpreter', 'latex');
if save_plot
    export_fig(['/Volumes/GoogleDrive/My Drive/Lior/Acoustics lab/Matlab/',...
        'Research/FB_BFBR/BSM/plots/numerical_errors/sparse/',...
        'norm_binaural_err_L=',num2str(L),'_',arrTypeTxt,'.png'], '-transparent', '-r300');
end
%}

%%Source estimation error plot
%{
figure;
c_order = num2cell(get(gca,'colororder'), 2);
sb = semilogx(freqs_sig, mag2db(err_s_BSM), 'linewidth', 3, 'linestyle', '-', 'DisplayName', 'BSM');
hold on;
ss = semilogx(freqs_sig, mag2db(err_s_sparse), 'linewidth', 3, 'linestyle', '-', 'DisplayName', sparse_method);
set(sb, {'color'}, c_order(1, :));
set(ss, {'color'}, c_order(2, :));
% ylim([-10 10]);
axis tight;
xlabel('Frequency [Hz]');
ylabel('Source Estimation Error [dB]');
title([num2str(L), ' sources, $\phi_{rot}$=',...
    num2str(rad2deg(head_rot_az)), '$^{\circ}$']);
legend('location', 'northwest', 'interpreter', 'latex');
% legend({'BSM', sparse_method}, 'interpreter', 'latex');
if save_plot
    export_fig(['/Volumes/GoogleDrive/My Drive/Lior/Acoustics lab/Matlab/',...
        'Research/FB_BFBR/BSM/plots/numerical_errors/sparse/',...
        'source_err_L=',num2str(L),'_',arrTypeTxt,'.png'], '-transparent', '-r300');
end
%}

%%Normalize Source estimation error plot
figure;
c_order = num2cell(get(gca,'colororder'), 2);
sb = semilogx(freqs_sig, mag2db(err_s_norm_BSM), 'linewidth', 3, 'linestyle', '-', 'DisplayName', 'BSM');
hold on;
ss = semilogx(freqs_sig, mag2db(err_s_norm_sparse), 'linewidth', 3, 'linestyle', '-', 'DisplayName', sparse_method);
set(sb, {'color'}, c_order(1, :));
set(ss, {'color'}, c_order(2, :));
% ylim([-10 10]);
axis tight;
xlabel('Frequency [Hz]');
ylabel('Normalized Source Estimation Error [dB]');
title([num2str(L), ' sources, $\phi_{rot}$=',...
    num2str(rad2deg(head_rot_az)), '$^{\circ}$']);
legend('location', 'northwest', 'interpreter', 'latex');
% legend({'BSM', sparse_method}, 'interpreter', 'latex');
if save_plot
    export_fig(['/Volumes/GoogleDrive/My Drive/Lior/Acoustics lab/Matlab/',...
        'Research/FB_BFBR/BSM/plots/numerical_errors/sparse/',...
        'norm_source_err_L=',num2str(L),'_',arrTypeTxt,'.png'], '-transparent', '-r300');
end
