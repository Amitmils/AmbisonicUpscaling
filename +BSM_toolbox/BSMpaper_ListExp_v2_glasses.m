%% This script generates binaural signals with BSM (complex and magnitude LS versions)

% Date created: November 24, 2020
% Created by:   Lior Madmoni
% Modified :    April 12, 2021    

clearvars;
close all;
clc;

restoredefaultpath;
% add ACLtoolbox path
addpath(genpath('/Users/liormadmoni/Google Drive/Lior/Acoustics lab/Matlab/Research/Github/general'));
addpath(genpath('/Users/liormadmoni/Google Drive/Lior/Acoustics lab/Matlab/Research/Github/MCRoomSim'));
cd('/Users/liormadmoni/Google Drive/Lior/Acoustics lab/Matlab/Research/Github/general');

startup_script();
rng('default');

% parameters/flags - array
filt_len = 0.032;                                      % filters (BSM/HRTF) length [sec]
head_rot_az = ...
    wrapTo2Pi(deg2rad([0, 30, 60]));                   % vector of head rotations [rad]
arrayTypeTxt = 'Glasses';
normSV = true;                                        % true - normalize steering vectors

% parameters/flags - general
c = 343;                                               % speed of sound [m/s]
desired_fs = 48000;                                    % choose samplong frequency in Hz
N_PW = 14;                                             % SH order of plane-wave synthesis

filt_samp    = filt_len * desired_fs;

% make sure filt_samp is even
if mod(filt_samp, 2)
    filt_samp = filt_samp + 1;
end

% parameters/flags - BSM design
BSM_inv_opt = 1;                                       % 1: ( (1 / lambda) * (A * A') + eye ),  2: ((A * A') + lambda * eye);
source_distribution = 1;                               % 0: nearly uniform (t-design), 1: spiral nearly uniform
Q = 240;                                               % Assumed number of sources
f_cut_magLS = 1500;                                    % cutoff frequency to use MagLS
tol_magLS = 1e-20;                                     % tolerance of iterative solution for MagLS
max_iter_magLS = 1E5;                                  % max number of iterations for MagLS
%noise related BSM parameters (regularization)
SNR = 20;                                              % assumed sensors SNR [dB]
sig_n = 0.1;
sig_s = 10^(SNR/10) * sig_n;
SNR_lin = sig_s / sig_n;    

% BSM grid
[th_BSMgrid_vec, ph_BSMgrid_vec] = BSM_toolbox.BSMgrid(source_distribution, Q);

% New CVX flag
magLS_cvx = false;

use_mcroomsim = true;
sig_idx = 3;                                           % 1: male speech, 2,3: female speech

%% Load Glasses ATF
atf_folder = '/Users/liormadmoni/Google Drive/Lior/Acoustics lab/Matlab/Research/FB_BFBR/Data/Glasses_array/';

Vt = h5read(fullfile(atf_folder,'Device_ATFs.h5'),'/IR');
Vt = permute(Vt,[1,3,2]);
th_atf = h5read(fullfile(atf_folder,'Device_ATFs.h5'),'/Theta');
ph_atf = h5read(fullfile(atf_folder,'Device_ATFs.h5'),'/Phi');
atf_fs = h5read(fullfile(atf_folder,'Device_ATFs.h5'),'/SamplingFreq_Hz');

if atf_fs ~= desired_fs
    Vt = resample(Vt, desired_fs, atf_fs, Dimension=2);
end

%Mics 5 and 6 are inside the ears - thus, remove them from simulation
Vt = Vt(1:4, :, :);

n_mic = size(Vt, 1);

% Vt is [mics x samples x measured directions]

%% ================= ATFs preprocessing
% Interpolate ATF to BSMGrid

% Convert ATF to SH domain
N_ATF = 14;
% Y_atf_pinv = pinv(shmat(N_ATF, [th_atf, ph_atf]));
Y_atf_before_pinv = shmat(N_ATF, [th_atf, ph_atf]);

[Vt_m, Vt_s, Vt_d] = size(Vt);
Vt_nm = reshape(Vt, [Vt_m * Vt_s, Vt_d]);
% Vt_nm = (Y_atf_pinv * Vt_nm.').';
Vt_nm = (Y_atf_before_pinv \ (Vt_nm.')).';

% Convert back to space domain in BSM grid
Y_atf = shmat(N_ATF, [th_BSMgrid_vec, ph_BSMgrid_vec]);
Vt_BSM = (Y_atf * Vt_nm.').';
Vt_BSM = reshape(Vt_BSM, [Vt_m, Vt_s, length(th_BSMgrid_vec)]);

% Vt_BSM is [mics x samples x BSM directions]

% Convert ATF to frequency domain
NFFT = filt_samp;
Vf_BSM = fft(Vt_BSM, NFFT, 2);

% remove negative frequencies
Vf_BSM(:, NFFT/2+1:end, :)=[];
freqs_atf = (0 : (size(Vf_BSM, 2) - 1))'*(desired_fs / NFFT);

% steering vector should be [n_mic x directions x freq] so reshape
Vf_BSM = permute(Vf_BSM, [1, 3, 2]);

fprintf('Finished ATF preprocessing\n');

%% ================= Room simulation

%signal
sig_paths = "/Users/liormadmoni/Google Drive/Lior/Acoustics lab/Matlab/" + ...
    "Research/FB_BFBR/Data/dry_signals/" + ["SI1138.wav",...
    "SI1392.wav",...
    "SI1461.wav"]; % male, female, female
sig_path = sig_paths(sig_idx);

if sig_idx == 1
    sig_gender = 'malespeech';
else
    sig_gender = 'femalespeech';
end

% sig_path = '/Users/liormadmoni/Google Drive/Lior/Acoustics lab/Matlab/Research/Github/general/+examples/data/female_speech.wav';  % location of .wav file - signal

[s, signal_fs] = audioread(sig_path);
if (signal_fs ~= desired_fs)
    s = resample(s, desired_fs, signal_fs);
    signal_fs = desired_fs;
end

%soundsc(s, desired_fs);
freqs_sig    = ( 0 : (filt_samp / 2) - 1 ) * desired_fs / filt_samp;
freqs_sig(1) = 1/4 * freqs_sig(2); %to not divide by zero

% Room params
switch sig_idx
    case 1
        R = 0.83; % walls refelection coeff - 0.7: T_60=0.19s, 0.92: T_60=0.75s
        roomDim = [10 6 3];
        arrayPos = [2 2 1.7];
        sourcePos = [5 4.5 1.7];
    case 2
        R = 0.86; % walls refelection coeff - 0.7: T_60=0.19s, 0.92: T_60=0.75s
        roomDim = [7 5 3];
        arrayPos = [3 2.5 1.7];
        sourcePos = [6 4 1.7];        
    case 3
        R = 0.92; % walls refelection coeff - 0.7: T_60=0.19s, 0.92: T_60=0.75s
        roomDim = [8 5 3];
        arrayPos = [4 4 1.7];
        sourcePos = [6 2 1.7];        
end

if use_mcroomsim
    % MCRoomSim
    Room = SetupRoom('Dim', roomDim, 'Absorption', (1 - R^2) * ones(6, 6));
    Sources = AddSource('Location', sourcePos, 'Type', sig_gender);
    Receivers = AddReceiver('Location', arrayPos, 'Type', 'sphharm', ...
        'MaxOrder', N_PW, 'ComplexSH', false);
    Options = MCRoomSimOptions('Fs', desired_fs);  % consider adding Duration
    RIR = RunMCRoomSim(Sources, Receivers, Room, Options);

    % Adjust SH orders according to our conventions:
    
    % Using GDrive ACLtoolbox
    P = MCRoomPerm(N_PW)';
    C = SHc2r(N_PW)';
    RIR = (C * P * RIR.').';

    % old adjustment
    %{

    %   1. order adjustments (ours: -n:1:n, MCRoomSim: n,-n,(n-1),-(n-1),...0)
    SH_ordering_adjust = [];
    for sh_order=0:N_PW
        sh_degrees = 2 * sh_order + 1;
        eye_sh_degree = eye(sh_degrees);        
        boaz_degree_convention = [2:2:sh_degrees, sh_degrees, ...
            (sh_degrees-2):-2:1];

        SH_ordering_adjust = blkdiag(SH_ordering_adjust,...
            eye_sh_degree(boaz_degree_convention, :));
    end
    
    %   2. theta degree convention: ours (Boaz book Fig. 1.1), MCRoomSim
    %   (See documentation in AddReceiver lines 124-125 for example)
    %   Thus, we need to use theta_ours = π/2 - theta_MCRoomSim and use the
    %   [NOT CORRECT!] Mirror symmetry along θ with respect to the equator, θ = π/2. See
    %   Boaz book equation (1.15)
    SH_theta_adjust = zeros((N_PW + 1)^2, 1);
    for sh_order=0:N_PW
        for sh_degree=-sh_order:1:sh_order
            SH_theta_adjust(sh_order^2 + sh_order + sh_degree + 1) = ...
                (-1)^(sh_order + sh_degree);
        end
    end
    SH_theta_adjust = diag(SH_theta_adjust);

    % RIR = (RIR * SH_ordering_adjust) * SH_theta_adjust;
    RIR = RIR * SH_ordering_adjust;
    % RIR = RIR * SH_theta_adjust;

    %}
        
    T60 = RoomParams.T60(RIR(:,1), desired_fs);
    fprintf('MCRoomSim T60 = %.2f sec\n', T60);
    % figure; plot((0:size(RIR,1)-1)/desired_fs, real(RIR(:,1))); xlabel('Time [sec]'); % plot the RIR of a00
    anm_t = fftfilt(RIR, s);

    % Spherical coordinates of direct sound 
    direct_sound_rel_cart = sourcePos - arrayPos;
    [th0, ph0, r0]=c2s(direct_sound_rel_cart(1), direct_sound_rel_cart(2), direct_sound_rel_cart(3));
    ph0 = mod(ph0, 2*pi);
    direct_sound_rel_sph = [r0, th0, ph0];
    disp(['Source position: (r,th,ph) = (' num2str(direct_sound_rel_sph(1),'%.2f') ','...
        num2str(direct_sound_rel_sph(2)*180/pi,'%.2f') ','...
        num2str(direct_sound_rel_sph(3)*180/pi,'%.2f') ')']);

else
    % Github ACL toolbox
    [hnm, parametric_rir] = image_method.calc_rir(desired_fs, roomDim, sourcePos, arrayPos, R, {}, {'array_type', 'anm', 'N', N_PW});
    T60 = RoomParams.T60(hnm(:,1), desired_fs);
    fprintf('Labs toolbox T60 = %.2f sec\n', T60);
    % figure; plot((0:size(hnm,1)-1)/desired_fs, real(hnm(:,1))); xlabel('Time [sec]'); % plot the RIR of a00
    anm_t = fftfilt(hnm, s);
    
    % Spherical coordinates of direct sound 
    direct_sound_rel_cart = parametric_rir.relative_pos(1, :);
    [th0, ph0, r0]=c2s(direct_sound_rel_cart(1), direct_sound_rel_cart(2), direct_sound_rel_cart(3));
    ph0 = mod(ph0, 2*pi);
    direct_sound_rel_sph = [r0, th0, ph0];
    disp(['Source position: (r,th,ph) = (' num2str(direct_sound_rel_sph(1),'%.2f') ','...
        num2str(direct_sound_rel_sph(2)*180/pi,'%.2f') ','...
        num2str(direct_sound_rel_sph(3)*180/pi,'%.2f') ')']);
end

%% ================= HRTFs preprocessing
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

fprintf('Finished HRTF preprocessing\n');

%% ================= Load WignerD Matrix
WignerDpath = '/Users/liormadmoni/Google Drive/ACLtoolbox/Data/WignerDMatrix_diagN=32.mat';
load(WignerDpath);
N_HRTF_rot = 30;
DN = (N_HRTF_rot + 1)^2; % size of the wignerD matrix
D_allAngles = D(:, 1 : DN);

%% ================== Create BSM struct
BSMobj.freqs_sig = freqs_sig;
BSMobj.N_PW = N_PW;    
BSMobj.c = c;
BSMobj.n_mic = n_mic;
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
BSMobj.Q = Q;
BSMobj.source_distribution = source_distribution;
BSMobj.desired_fs = desired_fs;
BSMobj.filt_samp = filt_samp;
BSMobj.magLS_cvx = magLS_cvx;

%% ================== Create Listening experiment struct
ListExp.sig_path = sig_path;
ListExp.N_ATF = N_ATF;
ListExp.N_PW = N_PW;
ListExp.roomDim = roomDim;
ListExp.arrayPos = arrayPos;
ListExp.sourcePos = sourcePos;
ListExp.use_mcroomsim = use_mcroomsim;
ListExp.n_mic = n_mic;
ListExp.N_HRTF = N_HRTF;
ListExp.T60 = T60;
ListExp.direct_sound_rel_sph = direct_sound_rel_sph;

%% ================= calculate array measurements - time domain (version 1)
%
% Convert anm_t to space domain with ATF directions
Y_a = shmat(N_PW, [th_atf, ph_atf]);
a_t = (Y_a * anm_t.').';
a_t = real(a_t);

% Convolove a_t with Vt (Vt is [mics x time samples x measured directions])

% zero-pad a_t recording to correct length
a_t = [a_t; zeros(filt_samp - 1, length(th_atf))];
p_array_t = zeros(size(a_t, 1), n_mic);
for m=1:n_mic
    p_array_t(:, m) = sum(fftfilt(squeeze(Vt(m, :, :)), a_t), 2);
end

% soundsc(real([p_array_t(:, 1).'; p_array_t(:, 4).']), desired_fs);
fprintf('Finished calculating array measurements\n');
%}

%% ================= calculate array measurements - freq domain - GO OVER BEFORE USING AGAIN!!!! THIS MAY NOT BE CORRECT !!!
%{
% Convert anm_t to space domain with ATF directions
Y_a = shmat(N_ATF, [th_atf, ph_atf]);
a_t = (Y_a * anm_t.').';

% === FFT

% pad with zeros to reduce time aliasing
c = soundspeed();
r_head = 0.1;
pad = ceil(r_head / c * desired_fs * 100); % r/c*100 is just a bound on the length of the impulse response of the system from anm to p.
a_t(end+1:end+1+pad, :) = 0; 

NFFT = 2^nextpow2(size(a_t,1));
a_f = fft(a_t, NFFT, 1);
Vf = fft(Vt, NFFT, 2);
% remove negative frequencies
a_f(NFFT/2+1:end, :)=[];
Vf(:, NFFT/2+1:end, :)=[];
freqs_array = (0 : (size(a_f,1) - 1))'*(desired_fs / NFFT);

% steering vector should be [f_len x directions x n_mic] so reshape
Vf = permute(Vf, [2, 3, 1]);

p_array_f = zeros(length(freqs_array), n_mic);
for m=1:n_mic
    p_array_f(:, m) = sum(squeeze(Vf(:, :, m)) .* a_f, 2);
end

% === IFFT

% pad negative frequencies with zeros (has no effect since we use ifft with
% "symmetric" flag
p_array_f(end+1:NFFT, :) = 0;
p_array_t = ifft(p_array_f, "symmetric");

% trim to size before power of 2 padding
p_array_t(size(a_t,1)+1:end,:) = [];

% soundsc(real([p_array_t(:, 1).'; p_array_t(:, 4).']), desired_fs);
fprintf('Finished calculating array measurements\n');

%}

%% ================= Binaural Reproduction
V_k = Vf_BSM;

for h=1:length(head_rot_az)
    %% ================= Interpolate HRTFs to BSM grid without head rotation (no compensation)
    hobj_NoRot = RotateHRTF(hobj_freq_grid, N_HRTF_rot, D_allAngles, 0);
    hobj_NoRot_BSM = hobj_NoRot;
    hobj_NoRot_BSM = hobj_NoRot_BSM.toSpace('SRC', th_BSMgrid_vec, ph_BSMgrid_vec);    
    
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
    [c_BSM_cmplx_NoComp_l, c_BSM_cmplx_NoComp_r] = BSM_toolbox.GenerateBSMfilters_faster(BSMobj, V_k, hobj_NoRot_BSM);
    
    % MagLS version
    BSMobj.magLS = true;
    [c_BSM_mag_l, c_BSM_mag_r] = BSM_toolbox.GenerateBSMfilters_faster(BSMobj, V_k, hobj_rot_BSM);
    [c_BSM_mag_NoComp_l, c_BSM_mag_NoComp_r] = BSM_toolbox.GenerateBSMfilters_faster(BSMobj, V_k, hobj_NoRot_BSM);
    
    %%======Post-processing BSM filters (time domain)
    [c_BSM_cmplx_l_time_cs, c_BSM_cmplx_r_time_cs] = ...
        BSM_toolbox.PostProcessBSMfilters(BSMobj, c_BSM_cmplx_l, c_BSM_cmplx_r);
    [c_BSM_cmplx_NoComp_l_time_cs, c_BSM_cmplx_NoComp_r_time_cs] = ...
        BSM_toolbox.PostProcessBSMfilters(BSMobj, c_BSM_cmplx_NoComp_l, c_BSM_cmplx_NoComp_r);
    [c_BSM_mag_l_time_cs, c_BSM_mag_r_time_cs] = ...
        BSM_toolbox.PostProcessBSMfilters(BSMobj, c_BSM_mag_l, c_BSM_mag_r);
    [c_BSM_mag_NoComp_l_time_cs, c_BSM_mag_NoComp_r_time_cs] = ...
        BSM_toolbox.PostProcessBSMfilters(BSMobj, c_BSM_mag_NoComp_l, c_BSM_mag_NoComp_r);
    
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
    c_BSM_cmplx_NoComp_l_time_cs = [c_BSM_cmplx_NoComp_l_time_cs(:, 1), c_BSM_cmplx_NoComp_l_time_cs(:, end:-1:2)];
    c_BSM_cmplx_NoComp_r_time_cs = [c_BSM_cmplx_NoComp_r_time_cs(:, 1), c_BSM_cmplx_NoComp_r_time_cs(:, end:-1:2)];
    c_BSM_mag_l_time_cs = [c_BSM_mag_l_time_cs(:, 1), c_BSM_mag_l_time_cs(:, end:-1:2)];
    c_BSM_mag_r_time_cs = [c_BSM_mag_r_time_cs(:, 1), c_BSM_mag_r_time_cs(:, end:-1:2)];
    c_BSM_mag_NoComp_l_time_cs = [c_BSM_mag_NoComp_l_time_cs(:, 1), c_BSM_mag_NoComp_l_time_cs(:, end:-1:2)];
    c_BSM_mag_NoComp_r_time_cs = [c_BSM_mag_NoComp_r_time_cs(:, 1), c_BSM_mag_NoComp_r_time_cs(:, end:-1:2)];
    % zero-pad array recording to correct length
    p_array_t_zp = [p_array_t; zeros(filt_samp - 1, n_mic)];
    %
    p_BSM_cmplx_t_l = (sum(fftfilt(c_BSM_cmplx_l_time_cs.', p_array_t_zp), 2));
    p_BSM_cmplx_t_r = (sum(fftfilt(c_BSM_cmplx_r_time_cs.', p_array_t_zp), 2));
    p_BSM_cmplx_NoComp_t_l = (sum(fftfilt(c_BSM_cmplx_NoComp_l_time_cs.', p_array_t_zp), 2));
    p_BSM_cmplx_NoComp_t_r = (sum(fftfilt(c_BSM_cmplx_NoComp_r_time_cs.', p_array_t_zp), 2));
    p_BSM_mag_t_l = (sum(fftfilt(c_BSM_mag_l_time_cs.', p_array_t_zp), 2));
    p_BSM_mag_t_r = (sum(fftfilt(c_BSM_mag_r_time_cs.', p_array_t_zp), 2));
    p_BSM_mag_NoComp_t_l = (sum(fftfilt(c_BSM_mag_NoComp_l_time_cs.', p_array_t_zp), 2));
    p_BSM_mag_NoComp_t_r = (sum(fftfilt(c_BSM_mag_NoComp_r_time_cs.', p_array_t_zp), 2));
    %}        
    
    fprintf('Finished BSM reproduction for head rotation idx = %d/%d\n'...
        , h, length(head_rot_az));
    
    %% ================= Ambisonics format reproduction of anm
    %%TODO: add equalization support
    headRotation = true; rotAngles = head_rot_az(h);
    N_REF = N_PW;
    N_ANCHOR = 1;

    DisplayProgress = true;
    ref_sig_rot_t = BinauralReproduction_from_anm(anm_t,...
        HRTFpath, desired_fs, N_REF, headRotation, rotAngles, WignerDpath);
    anchor_sig_rot_t = BinauralReproduction_from_anm(anm_t,...
        HRTFpath, desired_fs, N_ANCHOR, headRotation, rotAngles, WignerDpath);
    fprintf('Finished Ambisonics reproduction for head rotation idx = %d/%d\n'...
        , h, length(head_rot_az));


    %% ================= Prepare signals
    p_BSM_cmplx_Comp_t = cat(2, p_BSM_cmplx_t_l, p_BSM_cmplx_t_r);
    p_BSM_cmplx_NoComp_t = cat(2, p_BSM_cmplx_NoComp_t_l, p_BSM_cmplx_NoComp_t_r);
    p_BSM_mag_Comp_t = cat(2, p_BSM_mag_t_l, p_BSM_mag_t_r);
    p_BSM_mag_NoComp_t = cat(2, p_BSM_mag_NoComp_t_l, p_BSM_mag_NoComp_t_r);
    p_REF_t = ref_sig_rot_t;
    p_ANCHOR_t = anchor_sig_rot_t;

    %soundsc(p_BSM_cmplx_t, desired_fs);
    %soundsc(p_BSM_mag_t, desired_fs);
    %soundsc(p_REF_t, desired_fs);

    %% ================= Save audio
    output_path = sprintf("/Users/liormadmoni/Google Drive/Lior/Acoustics lab/Matlab/" + ...
        "Research/FB_BFBR/BSM/Journal_paper_ListExp_v2/MUSHRA_Matlab_template/" + ...
        "Signals/Signals/%s/sig=%d/",arrayTypeTxt, sig_idx);
    mkdir(output_path);
    save(strcat(output_path, "ListExpDetails.mat"), "ListExp");
    
    rot_str = sprintf("%d",round(rad2deg(head_rot_az(h))));
    output_path = strcat(output_path, "ph=", rot_str, "/");
    mkdir(output_path);

    % Reference signal
    p_ref_fileName = strcat(output_path, "ph=", rot_str, ...
        "_Ambisonics_N", sprintf("%d", N_REF),"_REF.wav");
    save_audio(p_ref_fileName, p_REF_t, desired_fs);

    % Anchor signal
    p_anchor_fileName = strcat(output_path, "ph=", rot_str, ...
        "_Ambisonics_N", sprintf("%d", N_ANCHOR),"_ANCHOR.wav");
    save_audio(p_anchor_fileName, p_ANCHOR_t, desired_fs);

    % cmplx LS - comp
    p_BSM_cmplx_comp_fileName = strcat(output_path, "ph=", rot_str, ...
        "_BSM_N", sprintf("%d_", N_PW), arrayTypeTxt, "_Comp_CmplxLS.wav");
    save_audio(p_BSM_cmplx_comp_fileName, p_BSM_cmplx_Comp_t, desired_fs);

    % cmplx LS - no comp
    p_BSM_cmplx_NoComp_fileName = strcat(output_path, "ph=", rot_str, ...
        "_BSM_N", sprintf("%d_", N_PW), arrayTypeTxt, "_NoComp_CmplxLS.wav");
    save_audio(p_BSM_cmplx_NoComp_fileName, p_BSM_cmplx_NoComp_t, desired_fs);

    % Mag LS - comp
    p_BSM_magLS_comp_fileName = strcat(output_path, "ph=", rot_str, ...
        "_BSM_N", sprintf("%d_", N_PW), arrayTypeTxt, "_Comp_MagLS.wav");
    save_audio(p_BSM_magLS_comp_fileName, p_BSM_mag_Comp_t, desired_fs);

    % Mag LS - no comp
    p_BSM_magLS_NoComp_fileName = strcat(output_path, "ph=", rot_str, ...
        "_BSM_N", sprintf("%d_", N_PW), arrayTypeTxt, "_NoComp_MagLS.wav");
    save_audio(p_BSM_magLS_NoComp_fileName, p_BSM_mag_NoComp_t, desired_fs);
    
    fprintf('Finished saving signals for head rotation idx = %d/%d\n'...
        , h, length(head_rot_az));
    
end

%% Utility functions
function save_audio(file_path, p, fs)
    p = 0.9 * p / max(max(abs(p)));
    audiowrite(file_path, p, fs);
end


%% Temporary internal functions for MCRoomSim
function perm=miniSHc2r(n)

    % a help function for SHc2r, permuting for each given n.

    perm = zeros((2*n+1));
    sizeP = size(perm,1);
    perm((floor(sizeP/2)+1),(floor(sizeP/2)+1)) = 1;
    for ii= 1:(floor(sizeP/2))
        perm((floor(sizeP/2)+1+ii),(floor(sizeP/2)+1+ii)) = 1/sqrt(2)*(-1)^ii;%*(-1)^ii;
        perm((floor(sizeP/2)+1+ii),(floor(sizeP/2)+1-ii)) = 1/sqrt(2);
        perm((floor(sizeP/2)+1-ii),(floor(sizeP/2)+1-ii)) = -1/(sqrt(2)*1j);%*(-1)^ii;
        perm((floor(sizeP/2)+1-ii),(floor(sizeP/2)+1+ii)) = +1/(sqrt(2)*1j)*(-1)^ii;
    end
end

function Perm=SHc2r(Nmax)

    % this code forms a permute matrix from the Normalized Complex Spherical Harmonics to
    % the Normalized Real Spherical Harmonics
    % Perm matrix hold the relation- Ynm_{Real} = Perm x Ynm_{Complex}

    Perm = zeros((Nmax+1)^2);
    sizeP = size(Perm,1);
    ind = 0;
    for n= 0:Nmax

        Perm((ind+1):(ind+1+(2*n+1)-1),(ind+1):(ind+1+(2*n+1)-1)) = miniSHc2r(n);
        ind = ind + (2*n +1);
    end

    Perm=conj(Perm);
   
end

function perm=miniMCRoomPerm(n)

    % a help function for MCRooPerm, permuting for each given n.

    perm = zeros((2*n+1));
    sizeP = size(perm,1);
    perm((floor(sizeP/2)+1),(2*n+1)) = 1;
    for ii= 1:(floor(sizeP/2))
        perm((floor(sizeP/2)+1-ii),(2*n+1) - 2*ii +1 ) = 1;
        perm((floor(sizeP/2)+1+ii),(2*n+1) - 2*ii ) = 1;
    end
end

function Perm=MCRoomPerm(Nmax)

    % this code forms a permute matrix that orders the coefficients that we use
    % to the order MCRoomSim does, following GenSHIndices.m The following does
    % so by C_{MCRoomSIM convention} = Perm x C_{our convention}

    Perm = zeros((Nmax+1)^2);
    sizeP = size(Perm,1);
    ind = 0;
    for n= 0:Nmax

        Perm((ind+1):(ind+1  +(2*n+1) - 1     ),(ind+1):(ind+1  +(2*n+1) - 1     )) = miniMCRoomPerm(n);
        ind = ind + (2*n +1);
    end

    Perm = inv(Perm); 
end






