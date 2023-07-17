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

% parameters/flags - general
c = 343;                                               % speed of sound [m/s]
desired_fs = 48000;                                   % choose samplong frequency in Hz
N_PW = 14;                                             % SH order of plane-wave synthesis
filt_len = 0.032;

%% MCRoomSim
roomDim = [10 6 3];
R = 0.92; % walls refelection coeff
Room = SetupRoom('Dim', roomDim, 'Absorption', (1 - R^2) * ones(6, 6));

sourcePos = [4 3.5 1.7] + [1 1 0];
Sources = AddSource('Location', sourcePos, 'Type', 'femalespeech');

arrayPos = [2 2 1.7];
Receivers = AddReceiver('Location', arrayPos, 'Type', 'sphharm', ...
    'MaxOrder', N_PW, 'ComplexSH', true);

Options = MCRoomSimOptions('Fs', desired_fs);  % consider adding Duration

RIR = RunMCRoomSim(Sources, Receivers, Room, Options);

mcroomsim_T60 = RoomParams.T60(RIR(:,1), desired_fs);
fprintf('MCRoomSim T60 = %.2f sec\n', mcroomsim_T60);
% figure; plot((0:size(RIR,1)-1)/desired_fs, real(RIR(:,1))); xlabel('Time [sec]'); % plot the RIR of a00

%% Labs-internal image method room
roomDim = [10 6 3];
sourcePos = [4 3.5 1.7] + [1 1 0];
arrayPos = [2 2 1.7];
R = 0.95; % walls refelection coeff
[hnm, parametric_rir] = image_method.calc_rir(desired_fs, roomDim, sourcePos, arrayPos, R, {}, {'array_type', 'anm', 'N', N_PW});
labs_T60 = RoomParams.T60(hnm(:,1), desired_fs);
fprintf('Labs simulation T60 = %.2f sec\n', labs_T60);

% Optional plot
% figure; plot((0:size(hnm,1)-1)/desired_fs, real(hnm(:,1))); xlabel('Time [sec]'); % plot the RIR of a00

% Spherical coordinates of direct sound 
direct_sound_rel_cart = parametric_rir.relative_pos(1, :);
[th0, ph0, r0]=c2s(direct_sound_rel_cart(1), direct_sound_rel_cart(2), direct_sound_rel_cart(3));
ph0 = mod(ph0, 2*pi);
direct_sound_rel_sph = [r0, th0, ph0];

disp(['Source position: (r,th,ph) = (' num2str(direct_sound_rel_sph(1),'%.2f') ','...
    num2str(direct_sound_rel_sph(2)*180/pi,'%.2f') ','...
    num2str(direct_sound_rel_sph(3)*180/pi,'%.2f') ')']);   

%% Filter signal with RIR

% Signal
%sig_path = '/Data/dry_signals/demo/SX293.WAV';
sig_path = '/Users/liormadmoni/Google Drive/Lior/Acoustics lab/Matlab/Research/Github/general/+examples/data/female_speech.wav';  % location of .wav file - signal
[s, desired_fs] = audioread(sig_path);
%soundsc(s, desired_fs);

% MCRoomSim RIR
filt_samp    = filt_len * desired_fs;
freqs_sig    = ( 0 : (filt_samp / 2) ) * desired_fs / filt_samp;
freqs_sig(1) = 1/4 * freqs_sig(2); %to not divide by zero
anm_mcroomsim_t = fftfilt(RIR, s);
%soundsc(real(anm_mcroomsim_t(:, 1)), desired_fs);

% Labs RIR
filt_samp    = filt_len * desired_fs;
freqs_sig    = ( 0 : (filt_samp / 2) ) * desired_fs / filt_samp;
freqs_sig(1) = 1/4 * freqs_sig(2); %to not divide by zero
anm_lab_t = fftfilt(hnm, s);
%soundsc(real(anm_lab_t(:, 1)), desired_fs);






