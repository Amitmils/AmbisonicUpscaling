%% 
% Author: Tom Shlomo, ACLab BGU, 2020
% 
% This example demonstrates how to:
%% 
% # Use the image method to generate an RIR.
% # Convolve a clean speech with an RIR to obtain plane wave decomposition signals 
% ("anm").
% # Calculate spatical correlation matrices (SCM) using a multichannel STFT 
% and frequency and time smoothing.
% # Calculate 2 kinds of bin-wise direct path dominance tests (DPD): the first 
% is based on how well the SCM is approximated by a rank 1 matrix. The second 
% is based on the similarity between the first eigenvector of the SCM to a steering 
% vector.
% # Estimate bin-wise DOA by applying MUSIC on the bins that passed the DPD 
% test.
% # Estimate a global DOA, by selecting the peak on the histogram of bin-wise 
% DOA estimates.
% # Steering a maximum directivity beamformer towards the estimated DOA, to 
% obtain a cleaner estimate of the speech signal.
% # Estimate the SNR of the estimated speech, using an SNR measure which is 
% delay and scale invariant.

% startup_script();
% rng('default');
% [s, fs] = audioread("+examples/data/female_speech.wav");
%% generate RIR and convolve with speech
function [s_hat, fs] = image_method_dpd_music_beamforming(s, fs,roomDim,sourcePos,arrayPos,R,N)

% N = 4;
% roomDim = [4 6 3];    
% sourcePos = [2 1 1.7]+0.1*randn(1,3);
% arrayPos = [2 5 1]+0.1*randn(1,3);
% R = 0.92; % walls refelection coeff
[hnm, parametric_rir] = image_method.calc_rir(fs, roomDim, sourcePos, arrayPos, R, {}, {"array_type", "anm", "N", N});
T60 = RoomParams.T60(hnm(:,1), fs);
fprintf("T60 = %.2f sec\n", T60);
figure; plot((0:size(hnm,1)-1)/fs, real(hnm(:,1))); % plot the RIR of a00
xlabel('Time [sec]');
anm = fftfilt(hnm, s); 
% soundsc(real(anm(:,1)), fs);
%% calculate STFT

windowLength_sec = 30e-3;
window = hann( round(windowLength_sec*fs) );
hop = floor(length(window)/4);
[anm_stft, f_vec, t_vec] = stft(anm, window, hop, [], fs);
anm_stft = anm_stft(1:size(anm_stft,1)/2+1,:,:); % discard negative frequencies
f_vec = f_vec(1:size(anm_stft,1));
significant_energy_mask = vecnorm(anm_stft, 2, 3) >= max(vecnorm(anm_stft, 2, 3), [], "all")*1e-3;

% plot
figure;
hIm = imagesc(t_vec, f_vec, 10*log10(abssq(anm_stft(:,:,1))), "AlphaData", 0.5+0.5*double(significant_energy_mask));
hIm.Parent.YDir = 'normal';
xlabel('Time [sec]'); ylabel('Frequency [Hz]');
axis tight;
ylim([0 6000]);
hIm.Parent.CLim = max(hIm.CData, [], 'all') + [-80 0]; % fix dynamic range of display
hold on;

freq_limits = [200, 5000];
freq_ind = f_vec >= freq_limits(1) & f_vec <= freq_limits(2);
significant_energy_mask = significant_energy_mask(freq_ind, :);
%% Perform smoothing and eigendecomposition

SCM = smoothing_stft(anm_stft(freq_ind,:,:), 'TimeSmoothingWidth', 4, 'FreqSmoothingWidth', 30, "UseTrace", true);
[U, lambda] = eignd(SCM, [3 4]);
first_eigenvec = reshape(U(:, :, :, 1), [], size(U,3));
%% rank based DPD

dpd_1_thresh = 0.3;
dpd_1= lambda(:,:,1)./sum(lambda, 3); % when dpd is close to 1, it means that SCM is nearly rank 1.
dpd_test_1 = dpd_1 >= dpd_1_thresh & significant_energy_mask;
fprintf("%d/%d bins passed the first DPD test\n", nnz(dpd_test_1), numel(dpd_test_1));
dpd_image = image(hIm.Parent, t_vec, f_vec(freq_ind), zeros([size(dpd_1) 3]), 'AlphaData', double(dpd_test_1));
dpd_test_1 = dpd_test_1(:);
%% eigenvector/steering vector simmilarity based DPD

dpd_2_thresh = 0.85;
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

beamformer_weights = shmat(N, doa_estimated_1);
s_hat = anm * beamformer_weights.';
s_hat = real(s_hat);
snr_before = calculate_snr(s, real(anm(:,1)));
snr_after = calculate_snr(s, s_hat);
fprintf("SNR before: %.1f dB\n", snr_before);
fprintf("SNR after: %.1f dB\n", snr_after);
fprintf("SNR gain: %.1f dB\n", snr_after-snr_before);
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