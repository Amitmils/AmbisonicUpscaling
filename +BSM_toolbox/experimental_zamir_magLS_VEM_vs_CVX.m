%% Compare MagLS VEM vs CVX method. Use measured ATF.

% Date created: March 18, 2021
% Created by:   Lior Madmoni

clearvars;
close all;
clc;

restoredefaultpath;
% add ACLtoolbox path
addpath(genpath('/Users/liormadmoni/Google Drive/Lior/Acoustics lab/Matlab/Research/Github/general'));
cd('/Users/liormadmoni/Google Drive/Lior/Acoustics lab/Matlab/Research/Github/general/');

startup_script();
rng('default');

% parameters/flags - array
filt_len = 0.032; %0.032;                              % filters (BSM/HRTF) length [sec]
array_rot_az = ...
    wrapTo2Pi(deg2rad([0]));                           % vector of head rotations [degrees]
normSV = true;                                         % true - normalize steering vectors

% parameters/flags - general
c = 343;                                               % speed of sound [m/s]
desired_fs = 48000;                                    % choose samplong frequency in Hz
saveFiles = false;                                     % save MATLAB files before time interpolation?

% parameters/flags - BSM design
inv_opt = 2;                                           % opt=1 -> ( (1 / lambda) * (A * A') + eye )  |||| opt2=1 -> ((A * A') + lambda * eye);
source_distribution = 1;                               % 0 - nearly uniform (t-design), 1 - spiral nearly uniform
Q = 240;                                               % Asuumed number of sources

magLS = true;                                          % true - magLS, false - complex LS
f_cut_magLS = 1500;                                    % above cutoff frequency use MagLS
tol_magLS = 1e-20;    
max_iter_magLS = 1E5;
%noise
SNR = 40;                                              % assumed sensors SNR [dB]
sig_n = 0.1;
sig_s = 10^(SNR/10) * sig_n;
SNR_lin = sig_s / sig_n;    
%signal
filt_samp     = filt_len * desired_fs;
freqs_sig    = ( 0 : (filt_samp / 2) ) * desired_fs / filt_samp;
freqs_sig(1) = 1/4 * freqs_sig(2); %to not divide by zero

% Text variables for plots 
if magLS    
    %LS_title = ['MagLS max-iter=',num2str(max_iter_number_magLS, '%1.1e'),' tol=',num2str(tol_magLS,'%1.1e')];
    LS_title = ['MagLS_fcut=',num2str(f_cut_magLS)];
else
    LS_title = 'CmplxLS';
end

%% ================= HRTFS preprocessing - with hObj
% load HRIRs
N_HRTF = 18;
% HRTFpath =  '/Users/liormadmoni/Google Drive/ACLtoolbox/Data/HRTF/earoHRIR_KU100_Measured_2702Lebedev.mat';
HRTFpath = '/Volumes/GoogleDrive/My Drive/Lior/Acoustics lab/Research/FB/Binaural_beamforming/Zamir problems/dataToShare/HATS052322_BK_earo.mat';
load(HRTFpath);         % hobj is HRIR earo object
hobj.shutUp = false;
[th_BSMgrid_vec, ph_BSMgrid_vec] = BSM_toolbox.BSMgrid(source_distribution, Q);
hobj = BSM_toolbox.HRTF2BSMgrid(hobj, N_HRTF, th_BSMgrid_vec, ph_BSMgrid_vec);

% Interpolate HRTF to frequencies
hobj_freq_grid = hobj;
if strcmp(hobj_freq_grid.dataDomain{1},'FREQ'), hobj_freq_grid=hobj_freq_grid.toTime(); end

% resample HRTF to desired_fs
hobj_freq_grid = hobj_freq_grid.resampleData(desired_fs);
hobj_freq_grid = hobj_freq_grid.toFreq(filt_samp);

% Trim negative frequencies
hobj_freq_grid.data = hobj_freq_grid.data(:, 1:ceil(filt_samp/2)+1, :);

%% Create BSM object to pass parameters to function more easily
BSMobj.freqs_sig = freqs_sig;
BSMobj.magLS = magLS;
BSMobj.f_cut_magLS = f_cut_magLS;
BSMobj.tol_magLS = tol_magLS;
BSMobj.max_iter_magLS = max_iter_magLS;
BSMobj.c = c;
BSMobj.th_BSMgrid_vec = th_BSMgrid_vec;
BSMobj.ph_BSMgrid_vec = ph_BSMgrid_vec;
BSMobj.normSV = normSV;
BSMobj.SNR_lin = SNR_lin;
BSMobj.inv_opt = inv_opt;
BSMobj.array_rot_az = array_rot_az;
BSMobj.Q = Q;
BSMobj.source_distribution = source_distribution;
BSMobj.desired_fs = desired_fs;
BSMobj.filt_samp = filt_samp;

%% ================= BSM filters calculation
for ar = 1 : length(array_rot_az)    
    tic;
    %% ================= Load ATFs
    ATFpath = '/Volumes/GoogleDrive/My Drive/Lior/Acoustics lab/Research/FB/Binaural_beamforming/Zamir problems/dataToShare/ATFs.mat';
    load(ATFpath); 
    if desired_fs ~= ATF.params.samplerate
        fprintf('Error in sampling frequency - please match sampling rate of HRTF, ATF and BSM filters!')
    end
    n_mic = size(ATF.IR, 1);
    BSMobj.n_mic = n_mic;

    %% ================= Generate BSM filters in frequency domain

    % steering vectors
    V_k = fft(ATF.IR, filt_samp, 3);
    V_k = V_k(:, :, 1:ceil(filt_samp/2)+1);
    % V_k dimensions are [n_mic x directions x f_len]
    
    % interpolate ATF directions to BSM grid
    N_SV = N_HRTF;
    V_k = permute(V_k, [2, 3, 1]);
    % V_k dimensions are [directions x f_len x n_mic]
    V_k_directions = size(V_k, 1);
    f_len = ceil(filt_samp / 2) + 1;
    V_k = reshape(V_k, [V_k_directions, f_len * n_mic]);
    
    Ynm = shmat(N_SV, [ATF.th, ATF.ph]);
    V_k_sh = pinv(Ynm) * V_k;
    V_k_interp = shmat(N_SV, [th_BSMgrid_vec, ph_BSMgrid_vec]) * V_k_sh;
    V_k_interp = reshape(V_k_interp, [Q, f_len, n_mic]);
    V_k_interp = permute(V_k_interp, [3, 1, 2]);
    % V_k_interp dimensions are [n_mic x directions x f_len]

    %=== BSM-MagLS with VEM
    BSMobj.magLS_cvx = false;   % true - solve as SDP with CVX toolbox, false - Variable Exchange Method
    [c_BSM_VEM_l_freq, c_BSM_VEM_r_freq] = BSM_toolbox.GenerateBSMfilters_faster(BSMobj, V_k_interp, hobj_freq_grid);
    
    %=== BSM-MagLS with CVX toolbox
    BSMobj.magLS_cvx = true;   % true - solve as SDP with CVX toolbox, false - Variable Exchange Method
    [c_BSM_CVX_l_freq, c_BSM_CVX_r_freq] = BSM_toolbox.GenerateBSMfilters_faster(BSMobj, V_k_interp, hobj_freq_grid);

    %% Post-processing BSM filters     
    [c_BSM_VEM_l_time_cs, c_BSM_VEM_r_time_cs] = ...
        BSM_toolbox.PostProcessBSMfilters(BSMobj, c_BSM_VEM_l_freq, c_BSM_VEM_r_freq);
    [c_BSM_CVX_l_time_cs, c_BSM_CVX_r_time_cs] = ...
        BSM_toolbox.PostProcessBSMfilters(BSMobj, c_BSM_CVX_l_freq, c_BSM_CVX_r_freq);
            
    % print status    
    if ar == 1
        t1 = toc;
        time_now = datetime('now');
        time_to_finish = time_now + seconds(t1 * (length(array_rot_az) - 1));    
        FinishTimeString = datestr(time_to_finish);
        
        fprintf('Time for one loop over head rotation is %.2f\n', t1);
        fprintf(['Estimated finish time is ',FinishTimeString,'\n\n']);
        fprintf('Finished BSM reproduction for head rotation index %d/%d', ar, length(array_rot_az));
    else
        fprintf(repmat('\b', 1, (length(num2str(length(array_rot_az))) + 1 + (length(num2str(ar - 1))))));
        fprintf('%d/%d',ar, length(array_rot_az));
    end
end
fprintf('\n');

%% plot BSM filters - VEM
% time domain left ear
figure();
subplot(2, 2, 1);
plot(linspace(0, filt_len, filt_samp), c_BSM_VEM_l_time_cs);
xlabel('Time [s]');
legend('mic1', 'mic2', 'mic3', 'mic4', 'mic5');
title('MagLS-BSM with VEM - left ear');

% magnitude response left ear
% figure;
subplot(2, 2, 3);
semilogx(freqs_sig, mag2db(abs(c_BSM_VEM_l_freq)));
xlabel('Frequency [Hz]');
% legend('mic1', 'mic2', 'mic3', 'mic4', 'mic5');
title('MagLS-BSM with VEM magnitude response - left ear');
ylim([-40, 30]);
xlim([0, 2*1e4]);

%time domain right ear
% figure;
subplot(2, 2, 2);
plot(linspace(0, filt_len, filt_samp), c_BSM_VEM_r_time_cs);
xlabel('Time [s]');
% legend('mic1', 'mic2', 'mic3', 'mic4', 'mic5');
title('MagLS-BSM with VEM - right ear');

% magnitude response right ear
% figure;
subplot(2, 2, 4);
semilogx(freqs_sig, mag2db(abs(c_BSM_VEM_r_freq)));
xlabel('Frequency [Hz]');
% legend('mic1', 'mic2', 'mic3', 'mic4', 'mic5');
title('MagLS-BSM with VEM magnitude response - right ear');
ylim([-40, 30]);
xlim([0, 2*1e4]);

%% plot BSM filters - CVX
% time domain left ear
figure();
subplot(2, 2, 1);
plot(linspace(0, filt_len, filt_samp), c_BSM_CVX_l_time_cs);
xlabel('Time [s]');
legend('mic1', 'mic2', 'mic3', 'mic4', 'mic5');
title('MagLS-BSM with CVX - left ear');

% magnitude response left ear
% figure;
subplot(2, 2, 3);
semilogx(freqs_sig, mag2db(abs(c_BSM_CVX_l_freq)));
xlabel('Frequency [Hz]');
% legend('mic1', 'mic2', 'mic3', 'mic4', 'mic5');
title('MagLS-BSM with CVX magnitude response - left ear');
ylim([-40, 30]);
xlim([0, 2*1e4]);

%time domain right ear
% figure;
subplot(2, 2, 2);
plot(linspace(0, filt_len, filt_samp), c_BSM_CVX_r_time_cs);
xlabel('Time [s]');
% legend('mic1', 'mic2', 'mic3', 'mic4', 'mic5');
title('MagLS-BSM with CVX - right ear');

% magnitude response right ear
% figure;
subplot(2, 2, 4);
semilogx(freqs_sig, mag2db(abs(c_BSM_CVX_r_freq)));
xlabel('Frequency [Hz]');
% legend('mic1', 'mic2', 'mic3', 'mic4', 'mic5');
title('MagLS-BSM with CVX magnitude response - right ear');
ylim([-40, 30]);
xlim([0, 2*1e4]);

%% Save mat files
if saveFiles
    savedir = ['/Volumes/GoogleDrive/My Drive/Lior/Acoustics lab/Research/FB/',...
        'Binaural_beamforming/Zamir problems/Lior analysis/'];
    
    time_now = datetime('now');
    DateString = datestr(time_now, 30);
    mkdir(savedir);
    
    % save only BSM output
    save([savedir,'/BSMfilters_',DateString,'.mat'], ...
        'c_BSM_VEM_l_time_cs', 'c_BSM_VEM_r_time_cs', ...
        'c_BSM_VEM_l_freq', 'c_BSM_VEM_r_freq', ...
        'c_BSM_CVX_l_time_cs', 'c_BSM_CVX_r_time_cs', ...
        'c_BSM_CVX_l_freq', 'c_BSM_CVX_r_freq', 'BSMobj');
end


