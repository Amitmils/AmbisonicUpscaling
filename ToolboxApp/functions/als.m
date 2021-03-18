    % Author: Tom Shlomo, ACLab BGU, 2020
function [x,fs] = als(s,fs,roomDim,sourcePos,arrayPos,R,K)

startup_script();
rng('default');
%[s, fs] = audioread("+examples/data/female_speech.wav");

%% generate RIR and convolve with speech
%roomDim = [7,5,3];
% sourcePos = [roomDim(1) * 2/3, roomDim(2)/2, 1.5] + rand_between([-0.5, 0.5], [1, 3]);
% arrayPos =  [roomDim(1) * 1/4, roomDim(2)/2, 1.5] + rand_between([-0.5, 0.5], [1, 3]);

% R = 0.95; % walls refelection coeff

[rir, parametric_rir] = image_method.calc_rir(fs, roomDim, sourcePos, arrayPos, R,...
    {"zero_first_delay", true}, {"array_type", "em32"}); %#ok<CLARRSTR>
p = fftfilt(rir, s);

%% apply ALS
% K = 20; % number of early reflections to consider
doa_noisy = randn_on_sphere(K, parametric_rir.omega(1:K, :), 5*pi/180); % noisy DOA with 5 deg std
delay_noisy = parametric_rir.delay(1:K) + [0; randn(K-1,1)] * 10e-6; % noisy delay with 10usec std
x_exp = parametric_rir.amp(1:K, :); % ground truth amplitudes

flim = [2000, 4000]; % frequency band for ALS to operate on
x = als.wrapper(p, fs, flim, doa_noisy, delay_noisy, "array_type", "em32", "plot_flag", 1, "x_exp", x_exp, "s_exp", s);

%% reconstruct the RIR using the estimated amplitudes and the noisy DOAs and delays
rir_reconstructed = image_method.rir_from_parametric(fs, delay_noisy, x, doa_noisy, "array_type", "em32");
figure; plot((0:size(rir, 1) - 1) / fs, rir(:, 1)); % plot the RIR of mic #1
xlabel('Time [sec]');

%% plot recostructed and original RIR (of the first microphone)
q = 1; % first microphone
% choose best scaling (for nicer visualization)
T = ceil(max(delay_noisy) * fs);
% scale = rir_reconstructed(1:T, q) \ rir(1:T, q);
scale = x_exp(1)/x(1);
figure("name", "RIR");
plot((0:size(rir, 1) - 1) / fs, rir(:, q), "DisplayName", "Original RIR");
hold on;
plot((0:size(rir_reconstructed, 1) - 1) / fs, rir_reconstructed(:, q) * scale, "DisplayName", "Reconstructed RIR");

xlabel('Time [sec]');
legend();

%% design an FIR raking filter
latency = round(20e-3*fs); % 20misec latency. higher is better performance
N = 1000; % raking filter length. higher is better performance, but slower
Q = size(p, 2);
L = size(rir_reconstructed, 1);
H = zeros(L + N - 1, N, Q);
for q = 1:Q
    H(:,:,q) = convmtx(rir_reconstructed(:, q), N);
end
H = reshape(H, size(H,1), []);
tol = norm(H, "fro") * 1e-3;
h = pinv(H, tol);
h = h(:, latency+1);
h = reshape(h, [], Q);

%%
total_responce = 0;
s_rake = 0;
for q = 1:Q
    total_responce = total_responce + fftfilt(h(:, q), rir(:,q));
    s_rake = s_rake + fftfilt(h(:, q), p(:, q));
end
figure; plot(total_responce);
x = [s; p(:,1); s_rake];
% soundsc([s; p(:,1); s_rake], fs);
end
