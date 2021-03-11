%% This script is an example for ACLtoolbox 
% 1. Simulate room (shoebox) using the image method in the SH domain
% 2. Generate array recordings
% 3. Perform PWD from array recordings (estimate anm)
% 4. DOA estimation with estimated anm

% References:
% [1] Rafaely, Boaz. "Fundamentals of spherical array processing". Vol. 8. Berlin: Springer, 2015.
% [2] Pulkki, Ville. "Parametric time-frequency domain spatial audio". Eds. Symeon Delikaris-Manias, and Archontis Politis. John Wiley & Sons, Incorporated, 2018.

% Date created: February 4, 2021
% Created by:   Lior Madmoni

% clearvars;
% close all;
% clc;
function [s_hat, fs] = pwd_dpd_music_beamforming(s, fs,roomDim,sourcePos,arrayPos,R,N_array,r_array)
startup_script();

%% ================= parameters/flags - general
c = soundspeed();               % speed of sound [m/s];                        
DisplayProgress = true;         % true: display progress on command window

%% ================= parameters/flags - spherical array
% N_array = 4;                    % SH order of array
% r_array = 0.042;                % array radius. 0.042 is similar to EM32 array
sphereType = "open";            % "open" / "rigid"

%================= generate spherical coordinates of spherical array   
[th_array, ph_array, weights_array] = sampling_schemes.t_design(N_array);                

%% ================= parameters/flags - source
N_PW = 15;                                      % SH order of plane-wave synthesis
% sig_path = "+examples/data/female_speech.wav";  % location of .wav file - signal

%% ================= parameters/flags - room
% roomDim =       [15.5 9.8 7.5];     % Room Dimensions (L,W,H) [m]
% sourcePos  =    [8.25 3.8 1.7];     % Source position (x,y,z) [m]
% arrayPos   =    [5 5 1.7];          % Receiver position (x,y,z) [m]
% R = 0.9;                            % walls refelection coeff

%% generate RIR and convolve with speech
% [s, fs] = audioread(sig_path);

[hnm, parametric_rir] = image_method.calc_rir(fs, roomDim, sourcePos, arrayPos, R, {}, {"array_type", "anm", "N", N_PW});
T60 = RoomParams.T60(hnm(:,1), fs); 
CriticalDist = RoomParams.critical_distance_diffuse(roomDim, R);
if DisplayProgress
    disp('Room parameters:');
    disp('================');
    fprintf("T60 = %.2f sec\n", T60);
    disp(['Critical distance = ' num2str(CriticalDist) ' m']);
end
figure; plot((0:size(hnm,1)-1)/fs, real(hnm(:,1))); % plot the RIR of a00
xlabel('Time [sec]');
anm_t = fftfilt(hnm, s); 
% soundsc(real(anm_t(:,1)), fs);

% transform to frequency domain
%{
NFFT = 2^nextpow2(size(anm_t, 1));
anm_f = fft(anm_t, NFFT, 1); 
% remove negative frequencies
anm_f = anm_f(1:NFFT/2+1, :);
% vector of frequencies
fVec = (0:NFFT-1)'/NFFT * fs;
fVec_pos = fVec(1 : NFFT/2 + 1);
%}

if DisplayProgress
    % Spherical coordinates of direct sound 
    direct_sound_rel_cart = parametric_rir.relative_pos(1, :);
    [th0, ph0, r0]=c2s(direct_sound_rel_cart(1), direct_sound_rel_cart(2), direct_sound_rel_cart(3));
    ph0 = mod(ph0, 2*pi);
    direct_sound_rel_sph = [r0, th0, ph0];
    
    disp(['Source position: (r,th,ph) = (' num2str(direct_sound_rel_sph(1),'%.2f') ','...
        num2str(direct_sound_rel_sph(2)*180/pi,'%.2f') ','...
        num2str(direct_sound_rel_sph(3)*180/pi,'%.2f') ')']);   
        
    fprintf('\n');
    disp('Plane-wave density dimensions:');
    disp('=============================');
    disp(['anm_t is of size (samples, (N_PW + 1)^2) = (' num2str(size(anm_t, 1),'%d') ', ' num2str(size(anm_t, 2),'%d') ')']);
end

%% ================= Calculate array measurements  
p_array_t = anm2p(anm_t(:, 1:(N_array + 1)^2), fs, r_array, [th_array, ph_array], sphereType);
% trim zeros at the end of anm_est_t
p_array_t = p_array_t(1:size(anm_t, 1), :);
% soundsc(real([p_array_t(:, 1).'; p_array_t(:, 2).']), fs); 

if DisplayProgress
    fprintf('\n');
    disp('Array recordings dimensions:');
    disp('===========================');
    disp(['p_array_t is of size (samples, mics) = (' num2str(size(p_array_t, 1),'%d') ', ' num2str(size(p_array_t, 2),'%d') ')']);
end

%% ================= Plane wave decomposition
snr_db = 40;
anm_est_t = p2anm(p_array_t, fs, [th_array, ph_array], r_array, sphereType, snr_db, N_array);
% trim zeros at the end of anm_est_t
anm_est_t = anm_est_t(1:size(p_array_t, 1), :);
% soundsc(real(anm_est_t(:,1)), fs);

%% calculate STFT
anm = anm_est_t;   % anm_est_t / anm_t

windowLength_sec = 30e-3;
window = hann( round(windowLength_sec*fs) );
hop = floor(length(window)/4);
[anm_stft, f_vec_stft, t_vec] = stft(anm, window, hop, [], fs);
anm_stft = anm_stft(1:size(anm_stft,1)/2+1,:,:); % discard negative frequencies
f_vec_stft = f_vec_stft(1:size(anm_stft,1));
significant_energy_mask = vecnorm(anm_stft, 2, 3) >= max(vecnorm(anm_stft, 2, 3), [], "all")*1e-3;

% plot
figure;
hIm = imagesc(t_vec, f_vec_stft, 10*log10(abssq(anm_stft(:,:,1))), "AlphaData", 0.5+0.5*double(significant_energy_mask));
hIm.Parent.YDir = 'normal';
xlabel('Time [sec]'); ylabel('Frequency [Hz]');
axis tight;
ylim([0 6000]);
hIm.Parent.CLim = max(hIm.CData, [], 'all') + [-80 0]; % fix dynamic range of display
hold on;

% f_upper = fVec_pos( find(kr > N_array, 1, 'first') - 1 );
freq_limits = [200, 5000];
freq_ind = f_vec_stft >= freq_limits(1) & f_vec_stft <= freq_limits(2);
significant_energy_mask = significant_energy_mask(freq_ind, :);
%% Perform smoothing and eigendecomposition

SCM = smoothing_stft(anm_stft(freq_ind,:,:), 'TimeSmoothingWidth', 4, 'FreqSmoothingWidth', 30, "UseTrace", true);
[U, lambda] = eignd(SCM, [3 4]);
first_eigenvec = reshape(U(:, :, :, 1), [], size(U,3));
%% rank based DPD

dpd_1_thresh = 0.35;
dpd_1= lambda(:,:,1)./sum(lambda, 3); % when dpd is close to 1, it means that SCM is nearly rank 1.
dpd_test_1 = dpd_1 >= dpd_1_thresh & significant_energy_mask;
fprintf("%d/%d bins passed the first DPD test\n", nnz(dpd_test_1), numel(dpd_test_1));
dpd_image = image(hIm.Parent, t_vec, f_vec_stft(freq_ind), zeros([size(dpd_1) 3]), 'AlphaData', double(dpd_test_1));
dpd_test_1 = dpd_test_1(:);
%% eigenvector/steering vector simmilarity based DPD

dpd_2_thresh = 0.9;
dpd_2 = sphere_max_abs(first_eigenvec.', "newtonFlag", false, "normalization", "rho"); % using Newton's method is more accurate, however it takes longer
% when dpd_2 is close to 1, the first eigenvector is simmilar to a steering
% vector.
dpd_test_2 = dpd_2 >= dpd_2_thresh & significant_energy_mask(:)';
fprintf("%d/%d bins passed the second DPD test\n", nnz(dpd_test_2), numel(dpd_test_2));
dpd_image.AlphaData = double(reshape(dpd_test_2, size(U, 1), size(U, 2)));
%% Estimate DOA using music

doa_ground_truth = parametric_rir.omega(1,:);
doa_estimated_1 = estimate_doa(first_eigenvec(dpd_test_1, :).', doa_ground_truth, "Rank Based DPD");
doa_estimated_2 = estimate_doa(first_eigenvec(dpd_test_2, :).', doa_ground_truth, "MUSIC Based DPD");
%% Maximum Directivity Beamforming

beamformer_weights = shmat(N_array, doa_estimated_1);
s_hat = anm * beamformer_weights.';
s_hat = real(s_hat);
snr_before = calculate_snr(s, real(anm(:,1)));
snr_after = calculate_snr(s, s_hat);
fprintf("SNR before: %.1f dB\n", snr_before);
fprintf("SNR after: %.1f dB\n", snr_after);
fprintf("SNR gain: %.1f dB\n", snr_after-snr_before);
% soundsc(real(anm(:, 1)), fs);
% soundsc(s_hat, fs);
%%
function doa_estimated = estimate_doa(u, doa_ground_truth, name)
    [rho, omega] = sphere_max_abs(u, "newtonFlag", true);
    [hist, bins] = sphere_hist(omega, "tol", 20*pi/180, "nbins", 1000);
    [~, k] = max(hist);
    in_bin = angle_between(omega, bins(k,:)) <= 20*pi/180;
    doa_estimated = mean_doa(omega(in_bin, :));
    fprintf("%s: DOA error = %.2f degrees\n", name, angle_between(doa_estimated, doa_ground_truth)*180/pi);
    
    figure;
    hammer.scatter(omega, [], 50, rho, '.', "DisplayName", "Estimates");
    hammer.plot(doa_ground_truth, [], 'rx', 'MarkerSize', 20, 'LineWidth', 3, "DisplayName", "Ground Truth");
    hammer.plot(doa_estimated, [], 'm+', 'MarkerSize', 20, 'LineWidth', 3, "DisplayName", "Estimate");
    legend();
    title(name);
end
function snr_db = calculate_snr(s, s_hat)
    rho = max(abs(xcorr(s, s_hat, "normalized")));
    snr = 1/(1-rho.^2);
    snr_db = 10*log10(snr);
end
end