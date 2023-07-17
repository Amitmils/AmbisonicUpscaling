%% This script generates binaural signals with BSM (complex and magnitude LS versions)

% Date created: November 24, 2020
% Created by:   Lior Madmoni
% Modified :    April 12, 2021    

% clearvars;
% close all;
% clc;

% restoredefaultpath;
% add ACLtoolbox path
% addpath(genpath('/Users/liormadmoni/Google Drive/Lior/Acoustics lab/Matlab/Research/Github/general'));
% cd('/Users/liormadmoni/Google Drive/Lior/Acoustics lab/Matlab/Research/Github/general/');

function [p_BSM_mag_t, fs] = BSM_script(anm_t, fs, arrayType, rigidArray,  r_array, M, HRTFpath, N_PW, headRotation, rot_idx)

startup_script();
rng('default');
%??? taken from pwd_binaural_reproduction.m of ToolboxApp

%???

% parameters/flags - array
filt_len = 0.032;                                      % filters (BSM/HRTF) length [sec]
% arrayType = 1;                                         % 0 - spherical array, 1 - semi-circular array, 2 - full-circular array
% rigidArray = 1;                                        % 0 - open array, 1 - rigid array
% M = 6;                                                 % number of microphones
normSV = true;                                         % true - normalize steering vectors
% r_array = 0.1;                                         % array radius

% choose rotation angles
rotAngles = deg2rad(0:1:359);
head_rot_az = ...
    wrapTo2Pi(rotAngles(rot_idx));                         % vector of head rotations [rad]


% parameters/flags - general
c = soundspeed();                                      % speed of sound [m/s]
%desired_fs = 48000;                                   % choose samplong frequency in Hz
% N_PW = 14;                                             % SH order of plane-wave synthesis

% parameters/flags - BSM design
BSM_inv_opt = 1;                                       % 1 - ( (1 / lambda) * (A * A') + eye ),  2 - ((A * A') + lambda * eye);
source_distribution = 1;                               % 0 - nearly uniform (t-design), 1 - spiral nearly uniform
Q = 240;                                               % Assumed number of sources
f_cut_magLS = 1500;                                    % cutoff frequency to use MagLS
tol_magLS = 1e-20;                                     % tolerance of iterative solution for MagLS
max_iter_magLS = 1E5;                                  % max number of iterations for MagLS
%noise related BSM parameters (regularization)
SNR = 20;                                              % assumed sensors SNR [dB]
sig_n = 0.1;
sig_s = 10^(SNR/10) * sig_n;
SNR_lin = sig_s / sig_n;    

% Text variables for plots 
if ~rigidArray
    sphereType = 'open';
else
    sphereType = 'rigid';
end
%{
switch arrayType 
    case 0
        arrayTypeTxt = [sphereType,'Spherical'];
    case 1
        arrayTypeTxt = [sphereType,'SemiCirc'];
    case 2
        arrayTypeTxt = [sphereType,'FullCirc'];
end
%}

[th_BSMgrid_vec, ph_BSMgrid_vec] = BSM_toolbox.BSMgrid(source_distribution, Q);

%% generate RIR and convolve with speech
%signal
%sig_path = '/Data/dry_signals/demo/SX293.WAV';
% sig_path = "/Users/liormadmoni/Google Drive/Lior/Acoustics lab/Matlab/Research/Github/general/+examples/data/female_speech.wav";  % location of .wav file - signal
% [s, desired_fs] = audioread(sig_path);
%soundsc(s, desired_fs);
filt_samp    = filt_len * fs;
freqs_sig    = ( 0 : (filt_samp / 2) ) * fs / filt_samp;
freqs_sig(1) = 1/4 * freqs_sig(2); %to not divide by zero
% room
% roomDim = [4 6 3];
% sourcePos = [2 1 1.7]+0.1*randn(1,3);
% arrayPos = [2 5 1]+0.1*randn(1,3);
% R = 0.92; % walls refelection coeff

%% ================= HRTFS preprocessing
% load HRIRs
% N_HRTF = 30;
% HRTFpath =  '/Users/liormadmoni/Google Drive/ACLtoolbox/Data/HRTF/earoHRIR_KU100_Measured_2702Lebedev.mat';
%HRTFpath =  '/Users/liormadmoni/Google Drive/ACLtoolbox/Data/HRTF/earoHRIR_KEMAR_TU_BEM_OnlyHead.mat';
load(HRTFpath);         % hobj is HRIR earo object
hobj.shutUp = false;
%%Interpolate HRTF to frequencies
hobj_freq_grid = hobj;
if strcmp(hobj_freq_grid.dataDomain{1},'FREQ'), hobj_freq_grid=hobj_freq_grid.toTime(); end
% resample HRTF to desired_fs
hobj_freq_grid = hobj_freq_grid.resampleData(fs);
hobj_freq_grid = hobj_freq_grid.toFreq(filt_samp);
% Trim negative frequencies
hobj_freq_grid.data = hobj_freq_grid.data(:, 1:ceil(filt_samp/2)+1, :);

%% ================= Load WignerD Matrix
WignerDpath = 'ToolboxApp/data/WignerDMatrix_diagN=32.mat';   % needed just for headRotation
load(WignerDpath);
N_HRTF_rot = 30;
DN = (N_HRTF_rot + 1)^2; % size of the wignerD matrix
D_allAngles = D(:, 1 : DN);
clear D
%% ==================Create BSM struct
BSMobj.freqs_sig = freqs_sig;
BSMobj.N_PW = N_PW;    
BSMobj.c = c;
BSMobj.r_array = r_array;
BSMobj.rigidArray = rigidArray;
BSMobj.th_BSMgrid_vec = th_BSMgrid_vec;
BSMobj.ph_BSMgrid_vec = ph_BSMgrid_vec;
%
BSMobj.f_cut_magLS = f_cut_magLS;
BSMobj.tol_magLS = tol_magLS;
BSMobj.max_iter_magLS = max_iter_magLS;
BSMobj.normSV = normSV;
BSMobj.SNR_lin = SNR_lin;
BSMobj.inv_opt = BSM_inv_opt;
BSMobj.head_rot_az = head_rot_az;
BSMobj.M = M;
BSMobj.Q = Q;
BSMobj.source_distribution = source_distribution;
BSMobj.desired_fs = fs;
BSMobj.filt_samp = filt_samp;
BSMobj.sphereType = sphereType;

for m = 1:length(M)
    %% ================= Get array positions
    n_mic = M(m);        
    [th_array, ph_array, ~] = BSM_toolbox.GetArrayPositions(arrayType, n_mic, 0);       
    
    %% ==================Update BSM struct
    BSMobj.n_mic = n_mic;
    BSMobj.th_array = th_array;
    BSMobj.ph_array = ph_array;      
    
    %% ================= calculate array measurements   
    N_SV = N_PW;
    p_array_t = anm2p(anm_t(:, 1:(N_SV + 1)^2), fs, r_array, [th_array.', ph_array.'], sphereType);
    % trim zeros at the end of anm_est_t
    p_array_t = p_array_t(1:size(anm_t, 1), :);
    %p_array_t = circshift(p_array_t, round((size(p_array_t, 1) - 1) / 2), 1);        
    % soundsc(real([p_array_t(:, 1).'; p_array_t(:, 20).']), desired_fs);
    fprintf('Finished calculating array measurements\n');
    
    %% ================= calculate array steering vectors (for BSM filters)    
    V_k = CalculateSteeringVectors(BSMobj, N_SV, th_BSMgrid_vec, ph_BSMgrid_vec); 
    V_k = permute(V_k, [3 2 1]);    
    
    for h=1:length(head_rot_az)
        %% ================= Rotate HRTFs according to head rotation - new        
        hobj_rot = RotateHRTF(hobj_freq_grid, N_HRTF_rot, D_allAngles, head_rot_az(h));
        % Interpolate HRTF to BSM grid
        hobj_rot_BSM = hobj_rot;
        hobj_rot_BSM = hobj_rot_BSM.toSpace('SRC', th_BSMgrid_vec, ph_BSMgrid_vec);  

        %% ================= BSM method
        %%======Generate BSM filters in frequency domain
        %BSMobj.ph_array = ph_rot_array;
        % Complex version
        BSMobj.magLS = false;
        [c_BSM_cmplx_l, c_BSM_cmplx_r] = BSM_toolbox.GenerateBSMfilters_faster(BSMobj, V_k, hobj_rot_BSM);
        
        % MagLS version
        BSMobj.magLS = true;
        [c_BSM_mag_l, c_BSM_mag_r] = BSM_toolbox.GenerateBSMfilters_faster(BSMobj, V_k, hobj_rot_BSM);
        
        %%======Post-processing BSM filters (time domain)
        [c_BSM_cmplx_l_time_cs, c_BSM_cmplx_r_time_cs] = ...
            BSM_toolbox.PostProcessBSMfilters(BSMobj, c_BSM_cmplx_l, c_BSM_cmplx_r);
        [c_BSM_mag_l_time_cs, c_BSM_mag_r_time_cs] = ...
            BSM_toolbox.PostProcessBSMfilters(BSMobj, c_BSM_mag_l, c_BSM_mag_r);
        
        %%======Optional - plot filters in freq domain   
        %BSM_toolbox.PlotBSMfilters(BSMobj, c_BSM_cmplx_l_time_cs, 'time');
        
        %%======Filter microphone signals        
        % Direct filtering in frequency domain
        %{
        nfft = filt_samp + size(p_array_t, 1) - 1;
        c_BSM_cmplx_l_time_cs_f = fft(c_BSM_cmplx_l_time_cs, nfft, 2);
        c_BSM_cmplx_r_time_cs_f = fft(c_BSM_cmplx_r_time_cs, nfft, 2);
        c_BSM_mag_l_time_cs_f = fft(c_BSM_mag_l_time_cs, nfft, 2);
        c_BSM_mag_r_time_cs_f = fft(c_BSM_mag_r_time_cs, nfft, 2);
        %
        p_array_f = fft(p_array_t, nfft, 1);

        p_tmp = sum(conj(c_BSM_cmplx_l_time_cs_f.') .* p_array_f, 2);
        p_BSM_cmplx_t_l = ifft(p_tmp, nfft, 1, 'symmetric');

        p_tmp = sum(conj(c_BSM_cmplx_r_time_cs_f.') .* p_array_f, 2);
        p_BSM_cmplx_t_r = ifft(p_tmp, nfft, 1, 'symmetric');

        p_tmp = sum(conj(c_BSM_mag_l_time_cs_f.') .* p_array_f, 2);
        p_BSM_mag_t_l = ifft(p_tmp, nfft, 1, 'symmetric');

        p_tmp = sum(conj(c_BSM_mag_r_time_cs_f.') .* p_array_f, 2);
        p_BSM_mag_t_r = ifft(p_tmp, nfft, 1, 'symmetric');        
        %}
                
        %%Direct filtering in time domain - using fftfilt
        %
        % Time reversal - to conjugate filters in frequency domain        
        c_BSM_cmplx_l_time_cs = [c_BSM_cmplx_l_time_cs(:, 1), c_BSM_cmplx_l_time_cs(:, end:-1:2)];
        c_BSM_cmplx_r_time_cs = [c_BSM_cmplx_r_time_cs(:, 1), c_BSM_cmplx_r_time_cs(:, end:-1:2)];
        c_BSM_mag_l_time_cs = [c_BSM_mag_l_time_cs(:, 1), c_BSM_mag_l_time_cs(:, end:-1:2)];
        c_BSM_mag_r_time_cs = [c_BSM_mag_r_time_cs(:, 1), c_BSM_mag_r_time_cs(:, end:-1:2)];               
        % zero-pad array recording to correct length
        p_array_t_zp = [p_array_t; zeros(filt_samp - 1, n_mic)];
        %
        p_BSM_cmplx_t_l = (sum(fftfilt(c_BSM_cmplx_l_time_cs.', p_array_t_zp), 2));
        p_BSM_cmplx_t_r = (sum(fftfilt(c_BSM_cmplx_r_time_cs.', p_array_t_zp), 2));
        p_BSM_mag_t_l = (sum(fftfilt(c_BSM_mag_l_time_cs.', p_array_t_zp), 2));
        p_BSM_mag_t_r = (sum(fftfilt(c_BSM_mag_r_time_cs.', p_array_t_zp), 2));                              
        %}        
        
        fprintf('Finished BSM reproduction for mic idx = %d/%d, head rotation idx = %d/%d\n'...
            ,m, length(M), h, length(head_rot_az));
        
    end
    
end
clear p_array_t p_array_t_zp V_k
%% Ambisonics format reproduction of anm
%%TODO: add equalization support
% headRotation = true; 
rotAngles = head_rot_az;
N_BR = 14;
% DisplayProgress = true;
% bin_sig_rot_t = BinauralReproduction_from_anm(anm_t,...
%     HRTFpath, fs, N_BR, headRotation, rotAngles, WignerDpath);
% 

%% Listen to results
% p_BSM_cmplx_t = cat(2, p_BSM_cmplx_t_l, p_BSM_cmplx_t_r);
p_BSM_mag_t = cat(2, p_BSM_mag_t_l, p_BSM_mag_t_r);
% p_REF_t = bin_sig_rot_t;
end
%soundsc(p_BSM_cmplx_t, desired_fs);
%soundsc(p_BSM_mag_t, desired_fs);
%soundsc(p_REF_t, desired_fs);















