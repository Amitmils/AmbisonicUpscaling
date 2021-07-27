%% This script generates binaural signals with BSM (complex and magnitude LS versions) with 18mics array

% Date created: June 21, 2021
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
filt_len = 0.032;                                      % filters (BSM/HRTF) length [sec]
arrayType = 1;                                         % 0 - 90 equiangle horizontal plane 1m, 1 - 110 equiangle horizontal plane 3m
head_rot_az = ...
    wrapTo2Pi(deg2rad([0]));                           % vector of head rotations [rad]
normSV = false;                                        % true - normalize steering vectors

% parameters/flags - general
c = 343;                                               % speed of sound [m/s]
%desired_fs = 48000;                                   % choose samplong frequency in Hz
N_PW = 14;                                             % SH order of plane-wave synthesis

% parameters/flags - BSM design
BSM_inv_opt = 1;                                       % 1 - ( (1 / lambda) * (A * A') + eye ),  2 - ((A * A') + lambda * eye);
f_cut_magLS = 1500;                                    % cutoff frequency to use MagLS
tol_magLS = 1e-20;                                     % tolerance of iterative solution for MagLS
max_iter_magLS = 1E4;                                  % max number of iterations for MagLS
%noise related BSM parameters (regularization)
SNR = 20;                                              % assumed sensors SNR [dB]
sig_n = 0.1;
sig_s = 10^(SNR/10) * sig_n;
SNR_lin = sig_s / sig_n;    

% Text variables for plots 
switch arrayType 
    case 0
        arrayTypeTxt = ['90 equiangle horizontal plane 1m'];
        arrayTypeLbl = ['18mic_90equiangle_1m'];
    case 1
        arrayTypeTxt = ['110 equiangle horizontal plane 3m'];
        arrayTypeLbl = ['18mic_110equiangle_3m'];
end

%% generate RIR and convolve with speech
%signal
%sig_path = '/Data/dry_signals/demo/SX293.WAV';
sig_path = "/Users/liormadmoni/Google Drive/Lior/Acoustics lab/Matlab/Research/Github/general/+examples/data/female_speech.wav";  % location of .wav file - signal
[s, desired_fs] = audioread(sig_path);
%soundsc(s, desired_fs);
filt_samp    = filt_len * desired_fs;
freqs_sig    = ( 0 : (filt_samp / 2) ) * desired_fs / filt_samp;
freqs_sig(1) = 1/4 * freqs_sig(2); %to not divide by zero
% room
roomDim = [6 6 3];
sourcePos = [4 1.5 1.7];    %+0.1*randn(1,3);
arrayPos = [2 3 1.7];       %+0.1*randn(1,3);
R = 0.96;                   % walls refelection coeff
[hnm, parametric_rir] = image_method.calc_rir(desired_fs, roomDim, sourcePos, arrayPos, R, {}, {"array_type", "anm", "N", N_PW});
T60 = RoomParams.T60(hnm(:,1), desired_fs);
fprintf("T60 = %.2f sec\n", T60);
% figure; plot((0:size(hnm,1)-1)/desired_fs, real(hnm(:,1))); xlabel('Time [sec]'); % plot the RIR of a00
anm_t = fftfilt(hnm, s);

%% ================= Load Array Data
arrayFold = '/Users/liormadmoni/Google Drive/Lior/Acoustics lab/Matlab/Research/FB_BFBR/Data/18mic_head_array/data_mesh2hrtf/';
switch arrayType
    case 0
        arrayData = load([arrayFold,'head_irs_circle_1m.mat']);
    case 1
        arrayData = load([arrayFold,'head_irs_circle_3m.mat']);
end
M = size(arrayData.irs, 2);

%resample IRs
if desired_fs ~= arrayData.fs
    arrayData.irs = resample(arrayData.irs, desired_fs, arrayData.fs);
    arrayData.fs = desired_fs;
end

% interpolate IRs to length fo filt_samp
arrayData.TFs = fft(arrayData.irs, filt_samp, 1);
%trim negative frequencies
arrayData.TFs = arrayData.TFs(1:ceil(filt_samp/2)+1, :, :);

th_BSMgrid_vec = arrayData.colatitude;
ph_BSMgrid_vec = arrayData.azimuth;

%% ================= HRTFS preprocessing
% load HRIRs
N_HRTF = 30;
HRTFpath =  '/Users/liormadmoni/Google Drive/ACLtoolbox/Data/HRTF/earoHRIR_KU100_Measured_2702Lebedev.mat';
%HRTFpath =  '/Users/liormadmoni/Google Drive/ACLtoolbox/Data/HRTF/earoHRIR_KEMAR_TU_BEM_OnlyHead.mat';
load(HRTFpath);         % hobj is HRIR earo object
hobj.shutUp = false;
%%Interpolate HRTF to frequencies
hobj_freq_grid = hobj;
if strcmp(hobj_freq_grid.dataDomain{1},'FREQ'), hobj_freq_grid=hobj_freq_grid.toTime(); end
% resample HRTF to desired_fs
hobj_freq_grid = hobj_freq_grid.resampleData(desired_fs);
hobj_freq_grid = hobj_freq_grid.toFreq(filt_samp);
% Trim negative frequencies
hobj_freq_grid.data = hobj_freq_grid.data(:, 1:ceil(filt_samp/2)+1, :);

%% ================= Load WignerD Matrix
WignerDpath = '/Users/liormadmoni/Google Drive/Lior/Acoustics lab/Matlab/Research/FB_BFBR/Data/WignerDMatrix_diagN=32.mat';
load(WignerDpath);
N_HRTF_rot = 30;
DN = (N_HRTF_rot + 1)^2; % size of the wignerD matrix
D_allAngles = D(:, 1 : DN);

%% ==================Create BSM struct
BSMobj.freqs_sig = freqs_sig;
BSMobj.N_PW = N_PW;    
BSMobj.c = c;
BSMobj.th_BSMgrid_vec = th_BSMgrid_vec;
BSMobj.ph_BSMgrid_vec = ph_BSMgrid_vec;
BSMobj.f_cut_magLS = f_cut_magLS;
BSMobj.tol_magLS = tol_magLS;
BSMobj.max_iter_magLS = max_iter_magLS;
BSMobj.normSV = normSV;
BSMobj.SNR_lin = SNR_lin;
BSMobj.inv_opt = BSM_inv_opt;
BSMobj.head_rot_az = head_rot_az;
BSMobj.M = M;
BSMobj.desired_fs = desired_fs;
BSMobj.filt_samp = filt_samp;
%%=Update BSM object 
BSMobj.n_mic = M;                
BSMobj.th_array = arrayData.colatitude;
BSMobj.ph_array = arrayData.azimuth;
 
%% ================= reshape array steering vectors according to BSM convention [n_mic x Q x freq]
V_k = permute(arrayData.TFs, [2 3 1]);    

%% ================= calculate array measurements  
% transform anm to a(theta,phi)
isComplexSH = true; transpose_flag = true;
Ynm = shmat(N_PW, [BSMobj.th_array.' BSMobj.ph_array.'], isComplexSH, transpose_flag);
a_t = anm_t * Ynm;
% zero-pad a(theta,phi) to correct length before filtering
a_t = [a_t; zeros(size(arrayData.irs, 1) - 1, size(arrayData.irs, 3))];
p_array_t = zeros(size(a_t, 1), M);
for m = 1:M
    p_array_t(:, m) = sum(fftfilt(squeeze(arrayData.irs(:, m, :)), a_t), 2);
end
% clear numerical errors
p_array_t = real(p_array_t);
% p_array_t = circshift(p_array_t, round((size(p_array_t, 1) - 1) / 2), 1);        
% soundsc(real([p_array_t(:, 1).'; p_array_t(:, 18).']), desired_fs);
fprintf('Finished calculating array measurements\n');

%% ================= BSM method
for h=1:length(head_rot_az)
    %% ================= Rotate HRTFs according to head rotation - new        
    hobj_rot = RotateHRTF(hobj_freq_grid, N_HRTF_rot, D_allAngles, head_rot_az(h));
    % Interpolate HRTF to BSM grid
    hobj_rot_BSM = hobj_rot;
    hobj_rot_BSM = hobj_rot_BSM.toSpace('SRC', th_BSMgrid_vec, ph_BSMgrid_vec);  
    
    %%======Generate BSM filters in frequency domain    
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
    p_array_t_zp = [p_array_t; zeros(filt_samp - 1, M)];
    %
    p_BSM_cmplx_t_l = (sum(fftfilt(c_BSM_cmplx_l_time_cs.', p_array_t_zp), 2));
    p_BSM_cmplx_t_r = (sum(fftfilt(c_BSM_cmplx_r_time_cs.', p_array_t_zp), 2));
    p_BSM_mag_t_l = (sum(fftfilt(c_BSM_mag_l_time_cs.', p_array_t_zp), 2));
    p_BSM_mag_t_r = (sum(fftfilt(c_BSM_mag_r_time_cs.', p_array_t_zp), 2));                              
    %}        

    fprintf('Finished BSM reproduction for head rotation idx = %d/%d\n'...
        , h, length(head_rot_az));

end
    

%% Ambisonics format reproduction of anm
%%TODO: add equalization support
headRotation = true; rotAngles = head_rot_az;
N_BR = 14;
DisplayProgress = true;
bin_sig_rot_t = BinauralReproduction_from_anm(anm_t,...
    HRTFpath, desired_fs, N_BR, headRotation, rotAngles, WignerDpath);


%% Listen to results
p_BSM_cmplx_t = cat(2, p_BSM_cmplx_t_l, p_BSM_cmplx_t_r);
p_BSM_mag_t = cat(2, p_BSM_mag_t_l, p_BSM_mag_t_r);
p_REF_t = bin_sig_rot_t;

%soundsc(p_BSM_cmplx_t, desired_fs);
%soundsc(p_BSM_mag_t, desired_fs);
%soundsc(p_REF_t, desired_fs);
















