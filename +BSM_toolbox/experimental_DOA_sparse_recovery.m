%% This script generates binaural signals with BSM (complex and magnitude LS versions)

% Date created: May 2021
% Created by:   Lior Madmoni

clearvars;
close all;
clc;

restoredefaultpath;
% add ACLtoolbox path
addpath(genpath('/Users/liormadmoni/Google Drive/Lior/Acoustics lab/Matlab/Research/Github/general'));
cd('/Users/liormadmoni/Google Drive/Lior/Acoustics lab/Matlab/Research/Github/general/');

% add sparse recovery scripts and CVX toolbox
addpath(genpath('/Volumes/GoogleDrive/My Drive/Lior/Acoustics lab/Matlab/Research/FB_BFBR/Sparse_recovery/l0_approximation/'));
addpath(genpath('/Volumes/GoogleDrive/My Drive/Lior/Acoustics lab/Matlab/Research/Solvers/CVX/'));

startup_script();
rng('default');

% parameters/flags - array
filt_len = 0.032;                                      % filters (BSM/HRTF) length [sec]
arrayType = 1;                                         % 0 - spherical array, 1 - semi-circular array, 2 - full-circular array, 3 - semi-circular array + cardioid mics near the ears
rigidArray = 1;                                        % 0 - open array, 1 - rigid array
M = 6;                                                 % number of microphones
r_array = 0.1;                                         % array radius
head_rot = ...
    wrapTo2Pi(deg2rad([0, 0]));                       % head position - (theta, phi) [rad]
normSV = true;                                         % true - normalize steering vectors
load_wigner = true;                                   % true - load matrix (good for azimuth rotation only), false - calculate wigner rotation matrix

% parameters/flags - general
c = 343;                                               % speed of sound [m/s]
desired_fs = 16000;                                   % choose samplong frequency in Hz
N_PW = 14;                                             % SH order of plane-wave synthesis

% parameters/flags - noise (regularization)
SNR = 80;                                              % assumed sensors SNR [dB]
sigma_s = 1;
sigma_n = 10^(-SNR/20) * sigma_s;
SNR_lin = (sigma_s / sigma_n)^2;  

% Sparse recovery parameters
omp_sigma = 1e-2;

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
switch arrayType 
    case 0
        arrayTypeTxt = [sphereType,'Spherical'];
    case 1
        arrayTypeTxt = [sphereType,'SemiCirc'];
    case 2
        arrayTypeTxt = [sphereType,'FullCirc'];
end

%% ================== Frequency vector
filt_samp    = filt_len * desired_fs;
freqs_sig    = ( 0 : (filt_samp / 2) ) * desired_fs / filt_samp;
freqs_sig(1) = 1/4 * freqs_sig(2); %to not divide by zero

% realizations
E = 2;

%% ================== Array parameters
n_mic = M(1);
[th_array, ph_array, ~] = BSM_toolbox.GetArrayPositions(arrayType, n_mic, 0);       

%% ================== Create BSM struct
BSMobj.freqs_sig = freqs_sig;
BSMobj.N_PW = N_PW;    
BSMobj.c = c;
BSMobj.r_array = r_array;
BSMobj.rigidArray = rigidArray;
BSMobj.normSV = normSV;
BSMobj.SNR_lin = SNR_lin;
BSMobj.M = M;
BSMobj.desired_fs = desired_fs;
BSMobj.filt_samp = filt_samp;
BSMobj.sphereType = sphereType;
BSMobj.n_mic = n_mic;
BSMobj.th_array = th_array;
BSMobj.ph_array = ph_array;      
    
%% ================= Sound field model
rng('default');
% DOAs
% doas_num = n_mic - 3;
doas_num = 2;
th_s = deg2rad(90 * ones(1, doas_num));
ph_s = deg2rad(linspace(0, 360 - 360 / (doas_num + 1), doas_num));

% steering vectors
N_SV = N_PW;
V_k = CalculateSteeringVectors(BSMobj, N_SV, th_s, ph_s); 
V_k = permute(V_k, [1 3 2]);

% noise
n = randn(filt_samp / 2 + 1, n_mic, E) + 1j * randn(filt_samp / 2 + 1, n_mic, E);
n = sigma_n * n ./ vecnorm(n, 2, 2);

% sound sources
% s = randn(filt_samp / 2 + 1, length(th_s), E) + 1j * randn(filt_samp / 2 + 1, length(th_s), E);
s = ones(filt_samp / 2 + 1, length(th_s), E) + 1j * ones(filt_samp / 2 + 1, length(th_s), E);
s = sigma_s * s ./ vecnorm(s, 2, 2);

% sound field model
p_f = zeros(filt_samp / 2 + 1, n_mic, E);
for f=1:(filt_samp / 2 + 1)
    V_f_norm = squeeze(V_k(f, :, :));
    p_f(f, :, :) = V_f_norm * squeeze(s(f, :, :)) + squeeze(n(f, :, :));
%     p_f(f, :, :) = V_f_norm * squeeze(s(f, :, :));
end

%% ================= Operating frequency and steering vectors
f_analysis = 500;
[~, f_ind] = min(abs(f_analysis - freqs_sig));

% steering vectors dictionary
[th_dict, ph_dict, ~] = sampling_schemes.equiangle(30); th_dict = th_dict.'; ph_dict = ph_dict.';
Q = length(th_dict);
V_dict = CalculateSteeringVectors(BSMobj, N_SV, th_dict, ph_dict);
V_dict = permute(V_dict, [3 2 1]);
V_dict_f = squeeze(V_dict(:, :, f_ind));

%%================= DOA est with sparse recovery
sparse_method = 'L1';

DOAs_OMP = [];
DOAs_IRLS = [];
DOAs_L1 = [];
sparse_time = [];
sparse_map = zeros(Q, E);

if strcmp(sparse_method, 'OMP')
    % OMP
    for e=1:E
        pf = squeeze(p_f(f_ind, :, e)).';
        pf = pf / vecnorm(pf);
        
        % Sparseland book code
        %
        
        % time
        tStart = tic;
        
        % OMP - Sparseland implementation
        [xOMP, choice, Sopt, Sopt_iter_order] = OMP(V_dict_f, pf, omp_sigma); 
        
        % timer end
        tEnd = toc(tStart);
        sparse_time = [sparse_time, tEnd];
        
        sparse_map(:, e) = abs(xOMP(:, choice)).^2;
        
        DOAs_OMP = [DOAs_OMP; th_dict(Sopt_iter_order), ph_dict(Sopt_iter_order)];
        %}
        
        % Matlab internal code
        %{
        
        % time
        tStart = tic;
        
        % OMP - MATLAB implementation
%         [coeff,dictatom,atomidx,errnorm] = ompdecomp(pf, V, 'MaxSparsity', 10);
        [coeff,dictatom,atomidx,errnorm] = ompdecomp(pf, V, 'MaxSparsity', n_mic);
        
        % timer end
        tEnd = toc(tStart);
        sparse_time = [sparse_time, tEnd - tStart];
        
        DOAs_OMP = [DOAs_OMP; th_dict(atomidx), ph_dict(atomidx)];
        %}
    end
    DOAs_est = DOAs_OMP;
elseif strcmp(sparse_method, 'IRLS')
    % IRLS
    for e=1:E
        pf = squeeze(p_f(f_ind, :, e)).';
        pf = pf / vecnorm(pf);
        
        % timer
        tStart = tic;
        
        % IRLS algorithm
        s_hat = IRLS_for_basisPursuit(V_dict_f, pf, ...
            irls_lambda, irls_delta, irls_thr, irls_print_iter);
        
        % timer end
        tEnd = toc(tStart);
        sparse_time = [sparse_time, tEnd];
        
        sparse_map(:, e) = abs(s_hat).^2;
        
        % DOA est
        IRLS_0_thr = 0.01;
        active_doa = abs(s_hat) > IRLS_0_thr;
        if sum(active_doa)
            DOAs_IRLS = [DOAs_IRLS; th_dict(active_doa), ph_dict(active_doa)];
        end
    end
    DOAs_est = DOAs_IRLS;
elseif strcmp(sparse_method, 'L1')
    % L1
    for e=1:E
        pf = squeeze(p_f(f_ind, :, e)).';
        pf = pf / vecnorm(pf);
        
        % timer
        tStart = tic;
        
        % CVX solver
        cvx_begin quiet
            variable alpha_hat(Q, 1) complex
            V_dict_f * alpha_hat == pf
            minimize(norm(alpha_hat, 1))
%             minimize(0.1 * norm(alpha_hat, 1) + 0.5 * norm(V * alpha_hat - pf))
        cvx_end
        
        % timer end
        tEnd = toc(tStart);
        sparse_time = [sparse_time, tEnd];
        
        s_hat = alpha_hat;
        sparse_map(:, e) = abs(s_hat).^2;
        
        % DOA est
        L1_0_thr = 0.01;
        active_doa = abs(s_hat) > L1_0_thr;
        if sum(active_doa)
            DOAs_L1 = [DOAs_L1; th_dict(active_doa), ph_dict(active_doa)];
        end
    end
    DOAs_est = DOAs_L1;
else
    % signal estimation with Tikhonov regularization
    fprintf('Sparse recovery method has to be one of the following: OMP/L1/IRLS!\n');
end

% print running time
fprintf('Average running time of %s algorithm is %f sec\n', sparse_method, mean(sparse_time));

%%== Scatter plots OMP DOAs
doas_num = size(DOAs_est, 1);
figure;
scatter(rad2deg(th_s), rad2deg(ph_s), 70, 'x', 'linewidth', 3, ...
    'DisplayName','GT');
hold on;
scatter(rad2deg(DOAs_est(:, 1)), rad2deg(DOAs_est(:, 2)), ...
    70, 'filled','linewidth', 3, 'DisplayName','est');
xlabel('$\theta$');
ylabel('$\phi$');
xlim([0 180]);
ylim([0 360]);
title(sprintf('%s, freq.=%d Hz', sparse_method, f_analysis));
legend;

%{
figure;
scatter(rad2deg(th_s), rad2deg(ph_s));
hold on;
scatter(rad2deg(th0) * ones(doas_num, 1),...
    rad2deg(DOAs_OMP(:, 2)) + randn(doas_num, 1));
xlabel('$\theta$');
ylabel('$\phi$');
%}

%% ================= Beamformers and sprase energy-map
%
%== DAS beamformer
DAS = zeros(Q, E);
for e=1:E
    DAS(:, e) = 1 / n_mic * V_dict_f' * (squeeze(p_f(f_ind, :, e)).');
end
DAS = abs(DAS).^2;

%== MVDR beamformer
Rn_hat = 1 / SNR_lin * eye(n_mic);
MVDR = zeros(Q, E);
% w_MVDR = (V' * (Rn_hat^(-1)) * V)^-1 * V' * (Rn_hat^(-1));
w_MVDR = pinv(V_dict_f' * V_dict_f) * V_dict_f';
for e=1:E    
    MVDR(:, e) = w_MVDR * (squeeze(p_f(f_ind, :, e)).');
end
MVDR = abs(MVDR).^2;

%== BSM beamformer
BSM_beam = zeros(Q, E);
for e=1:E
    BSM_beam(:, e) = V_dict_f' * pinv(V_dict_f * V_dict_f' + 1 / SNR_lin * eye(n_mic)) * (squeeze(p_f(f_ind, :, e)).');
end
BSM_beam = abs(BSM_beam).^2;

%%== Surface plot DAS beamformer
th_dict_unique = unique(th_dict);
ph_dict_unique = unique(ph_dict);
DAS_grid = zeros(length(th_dict_unique), length(ph_dict_unique), E);
for e=1:E
    DAS_grid(:, :, e) = reshape(DAS(:, e), length(th_dict_unique), length(ph_dict_unique));
end
figure;
surf(rad2deg(th_dict_unique), rad2deg(ph_dict_unique), DAS_grid(:, :, 1),...
    'EdgeColor', 'none');
hold on;
view(2);
axis tight;
shading interp;
plot3(rad2deg(th_s), rad2deg(ph_s), max(max(DAS_grid(:, :, 1))) * ones(1, length(th_s)),...
    'kx');
xlabel('$\theta$');
ylabel('$\phi$');
title(sprintf('DAS, freq.=%d Hz', f_analysis));
colorbar;

%%== Surface plot MVDR beamformer
th_dict_unique = unique(th_dict);
ph_dict_unique = unique(ph_dict);
MVDR_grid = zeros(length(th_dict_unique), length(ph_dict_unique), E);
for e=1:E
    MVDR_grid(:, :, e) = reshape(MVDR(:, e), length(th_dict_unique), length(ph_dict_unique));
end
figure;
surf(rad2deg(th_dict_unique), rad2deg(ph_dict_unique), MVDR_grid(:, :, 1),...
    'EdgeColor', 'none');
hold on;
view(2);
axis tight;
shading interp;
plot3(rad2deg(th_s), rad2deg(ph_s), max(max(MVDR_grid(:, :, 1))) * ones(1, length(th_s)),...
    'kx');
xlabel('$\theta$');
ylabel('$\phi$');
title(sprintf('MVDR, freq.=%d Hz', f_analysis));
colorbar;

%%== Surface plot BSM beamformer
th_dict_unique = unique(th_dict);
ph_dict_unique = unique(ph_dict);
BSM_beam_grid = zeros(length(th_dict_unique), length(ph_dict_unique), E);
for e=1:E
    BSM_beam_grid(:, :, e) = reshape(BSM_beam(:, e), length(th_dict_unique), length(ph_dict_unique));
end
figure;
surf(rad2deg(th_dict_unique), rad2deg(ph_dict_unique), BSM_beam_grid(:, :, 1),...
    'EdgeColor', 'none');
hold on;
view(2);
axis tight;
shading interp;
plot3(rad2deg(th_s), rad2deg(ph_s), max(max(BSM_beam_grid(:, :, 1))) * ones(1, length(th_s)),...
    'kx');
xlabel('$\theta$');
ylabel('$\phi$');
title(sprintf('BSM beam, freq.=%d Hz', f_analysis));
colorbar;

%%== Surface plot sparse recovery
th_dict_unique = unique(th_dict);
ph_dict_unique = unique(ph_dict);
sparse_map_grid = zeros(length(th_dict_unique), length(ph_dict_unique), E);
for e=1:E
    sparse_map_grid(:, :, e) = reshape(sparse_map(:, e), length(th_dict_unique), length(ph_dict_unique));
end
figure;
surf(rad2deg(th_dict_unique), rad2deg(ph_dict_unique), sparse_map_grid(:, :, 1),...
    'EdgeColor', 'none');
hold on;
view(2);
axis tight;
shading interp;
plot3(rad2deg(th_s), rad2deg(ph_s), max(max(sparse_map_grid(:, :, 1))) * ones(1, length(th_s)),...
    'kx');
xlabel('$\theta$');
ylabel('$\phi$');
title(sprintf('%s maps, freq.=%d Hz', sparse_method, f_analysis));
colorbar;

%}