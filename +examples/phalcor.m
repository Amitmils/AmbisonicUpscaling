% Author: Tom Shlomo, ACLab BGU, 2020

startup_script();
rng('default');
[s, fs] = audioread("+examples/data/female_speech.wav");

%% generate RIR and convolve with speech
room_size = "medium";
N = 4;
switch room_size
    case "medium"
        roomDim = [7,5,3];
        sourcePos = [roomDim(1)*2/3 roomDim(2)/2 1.5]+rand_between([-0.5, 0.5], [1, 3]);
        arrayPos =  [roomDim(1)*1/4 roomDim(2)/2 1.5]+rand_between([-0.5, 0.5], [1, 3]);
    case "small"
        roomDim = [4 6 3];
        sourcePos = [2 1 1.7]+0.1*randn(1,3);
        arrayPos = [2 5 1]+0.1*randn(1,3);
end
R = 0.9; % walls refelection coeff

[hnm, parametric_rir] = image_method.calc_rir(fs, roomDim, sourcePos, arrayPos, R, ...
    {"angle_dependence", false}, {"array_type", "anm", "N", N});
T60 = RoomParams.T60(hnm(:,1), fs);
fprintf("T60 = %.2f sec\n", T60);
figure; plot((0:size(hnm,1)-1)/fs, real(hnm(:,1))); % plot the RIR of a00
xlabel('Time [sec]');
anm = fftfilt(hnm, s);

%% calculate STFT
windowLength_sec = 150e-3;
window = hann( round(windowLength_sec*fs) );
hop = floor(length(window)/4);
[anm_stft, f_vec, t_vec] = stft(anm, window, hop, [], fs);
anm_stft = anm_stft(1:size(anm_stft,1)/2+1,:,:); % discard negative frequencies

%% apply PHALCOR
[estimates_global, estimates_local, expected, phalcor_intermediate_variables, hyperparams] = ...
    phalcor.wrapper(anm_stft, f_vec, t_vec, windowLength_sec,...
    "expected", parametric_rir, "taumax", 20e-3, "densityThresh", 0.03, ...
    "fine_delay_flag", 1, ...
    "plotFlag", 1, "intermediate_variables", struct());
% you can use the intermdiate variables to run PHALCOR from a mid-step.
% This is useful for faster hyperparameter tuning.
% If you want to use the cached variables to skip the tau detection step,
% use:
% "intermediate_variables", phalcor_intermediate_variables.T
%
% If you want to use the cached variables to skip to the clustering step,
% use:
% "intermediate_variables", phalcor_intermediate_variables.T2
%
% The latter is the most common case since the clustering algorithm is
% pretty sensitive to its hyper-parameters. specifically, tune
% "densityThresh" (between 0 and 1) to control how many bin-wise estimates
% are classified as outliers by dbscan. 0 is for none (not recommened).
