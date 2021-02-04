%% This script is an example for ACLtoolbox 
% 1. Simulate room (shoebox) using the image method in the SH domain
% 2. Generate array recordings
% 3. Perform PWD from array recordings (estimate anm)
% 4. Generate binaural signals (Ambisonics) from anm

% References:
% [1] Rafaely, Boaz. "Fundamentals of spherical array processing". Vol. 8. Berlin: Springer, 2015.
% [2] Pulkki, Ville. "Parametric time-frequency domain spatial audio". Eds. Symeon Delikaris-Manias, and Archontis Politis. John Wiley & Sons, Incorporated, 2018.
% [3] Rafaely, Boaz, and Amir Avni. "Interaural cross correlation in a sound field represented by spherical harmonics." The Journal of the Acoustical Society of America 127.2 (2010): 823-828.

% Date created: January 20, 2021
% Created by:   Lior Madmoni
% Modified:     February 4, 2021

clearvars;
close all;
clc;

startup_script();

%% ================= IMPORTANT!!! add path to HRTF and WignerD database from GoogleDrive ACLToolbox(in hobj format)
HRTFpath = '/Users/liormadmoni/Google Drive/ACLtoolbox/Data/HRTF/earoHRIR_KU100_Measured_2702Lebedev.mat';  
WignerDpath = '/Users/liormadmoni/Google Drive/ACLtoolbox/Data/WignerDMatrix_diagN=32.mat';   % needed just for headRotation

%% ================= parameters/flags - general
c = soundspeed();               % speed of sound [m/s]
DisplayProgress = true;         % true: display progress on command window

%% ================= parameters/flags - spherical array
N_array = 4;                    % SH order of array
r_array = 0.042;                % array radius. 0.042 is similar to EM32 array
sphereType = "open";            % "open" / "rigid"

%================= generate spherical coordinates of spherical array   
[th_array, ph_array, weights_array] = sampling_schemes.t_design(N_array);                

%% ================= parameters/flags - source
N_PW = 15;                                      % SH order of plane-wave synthesis
sig_path = "+examples/data/female_speech.wav";  % location of .wav file - signal

%% ================= parameters/flags - room
roomDim =       [15.5 9.8 7.5];     % Room Dimensions (L,W,H) [m]
sourcePos  =    [8.25 7.8 1.7];     % Source position (x,y,z) [m]
arrayPos   =    [5 5 1.7];          % Receiver position (x,y,z) [m]
R = 0.9;                            % walls refelection coeff

%% ================= parameters/flags - binaural reproduction
anm_to_reproduce = "est";       % binaural reproduction of ("sim": simulated anm, "est": estimated anm)
if strcmp(anm_to_reproduce, "sim")
    N_BR = N_PW;                % SH order of Ambisonics signal
elseif strcmp(anm_to_reproduce, "est")
    N_BR = N_array;             % SH order of Ambisonics signal
else
    error("Not a valid anm type for reproduction");
end
headRotation = true;            % true: generate rotated version of anm over azimuth - useful for head-tracking applications

%% generate RIR and convolve with speech
[s, fs] = audioread(sig_path);

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
NFFT = 2^nextpow2(size(anm_t, 1));
anm_f = fft(anm_t, NFFT, 1); 
% remove negative frequencies
anm_f = anm_f(1:NFFT/2+1, :);
% vector of frequencies
fVec = (0:NFFT-1)'/NFFT * fs;
fVec_pos = fVec(1 : NFFT/2 + 1);

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
    disp(['anm_f is of size (freq, (N_PW + 1)^2) = (' num2str(size(anm_f, 1),'%d') ', ' num2str(size(anm_f, 2),'%d') ')']);
end

%% ================= Calculate array measurements  
p_array_t = anm2p(anm_t(:, 1:(N_array + 1)^2), fs, r_array, [th_array, ph_array], sphereType);
% soundsc(real([p_array_t(:, 1).'; p_array_t(:, 2).']), fs); 
p_array_f = fft(p_array_t, NFFT, 1);
% remove negative frequencies
p_array_f = p_array_f(1:NFFT/2+1, :);

if DisplayProgress
    fprintf('\n');
    disp('Array recordings dimensions:');
    disp('===========================');
    disp(['p_array_f is of size (freq, mics) = (' num2str(size(p_array_f, 1),'%d') ', ' num2str(size(p_array_f, 2),'%d') ')']);
end

%% ================= Plane wave decomposition 
% Performing Tikhonov regularization for robust PWD as in [2] eq.(2.18)
Ymic = shmat(N_array, [th_array, ph_array]);
reg_meth = 1;
N_Bm = N_array;                                                 % SH order of plane-wave decomposition by Bn "inversion"
kr = 2 * pi * fVec_pos * r_array / c;
Bm2 = bn(N_Bm, kr, "sphereType", sphereType);

switch reg_meth
    case 1
        % same regularization for all frequencies
        lambda_tikhonov = db2mag(-50);                                   % regularization term
        Bm_reg = conj(Bm2) ./ (abs(Bm2).^2 + lambda_tikhonov^2);         % eq.(2.18) in [2]
    case 2
        % frequency dependent regularization              
        lambda_tikhonov = db2mag(-50);                                   % regularization term                
        Bm_reg_den = abs(Bm2).^2;
        Bm_reg_den(Bm_reg_den < lambda_tikhonov^2) = lambda_tikhonov^2;        
        Bm_reg = conj(Bm2) ./ Bm_reg_den;                                
end
%figure; semilogx(fVec_pos, 10*log10(abs(Bm_reg))); xlabel('Frequency [Hz]'); ylabel('Magntitude [dB]');

% SFT of array recordings 
pnm_f = (pinv(Ymic) * p_array_f.').';
% Perform plane-wave decomposition in the frequency domain
anm_est_f = Bm_reg .* pnm_f;

% Transform anm_est_f to time domain
% pad negative frequencies with zeros (has no effect since we use ifft with "symmetric" flag)
anm_est_f(end+1:NFFT, :) = 0;
anm_est_t = ifft(anm_est_f, [], 1, 'symmetric');
% trim to size before power of 2 padding
anm_est_t(size(anm_t,1)+1:end,:) = [];
% soundsc([real(anm_t(:, 1))], fs); 
% soundsc([real(anm_est_t(:, 1))], fs); 


%% ================= Generate binaural signals - Ambisonics format
if DisplayProgress
    fprintf('\n');
    disp('Binaural reproduction calculations:');
    disp('==================================');
end

%=============== Choose which anm to use for binaural reproduction: simulated or estimated from array
if strcmp(anm_to_reproduce, "sim")
    anm_BR = anm_f.'; 
    anm_BR = [anm_BR, conj(anm_BR(:, end-1:-1:2))];  % just to be consistent size-wise
elseif strcmp(anm_to_reproduce, "est")
    anm_BR = anm_est_f.';       
end

% load HRTF to an hobj struct -
load(HRTFpath);                     % hobj is HRIR earo object - domains are given in hobj.dataDomain
hobj.shutUp = ~DisplayProgress;     % shutUp parameter of hobj
if DisplayProgress    
    disp(['After loading HRTF, the dimensions are: (',hobj.dataDesc,')']);
end

% resample HRTF to desired_fs
if strcmp(hobj.dataDomain{1},'FREQ'), hobj=hobj.toTime(); end
if hobj.fs ~= fs
    [P_rat,Q_rat] = rat(fs / hobj.fs);
    hrir_l = hobj.data(:, :, 1).';
    hrir_r = hobj.data(:, :, 2).';
    hrir_l = resample(hrir_l, double(P_rat), double(Q_rat)).';
    hrir_r = resample(hrir_r, double(P_rat), double(Q_rat)).';

    hobj.data = cat(3, hrir_l, hrir_r);     
    hobj.fs = fs;        
end


% General function to generates binaural signals with or without head rotations over azimuth
% according to [3] eq. (9)
[bin_sig_rot_t, rotAngles] = BinSigGen_HeadRotation_ACL(hobj, anm_BR(1:(N_BR+1)^2, :), N_BR, headRotation, WignerDpath);
% *** NOTE: it is much more efficient to use RIR-anm instead of signals containing
% anm, but this is an example for binaural reproduction from estimated anm.
% If RIR is given, use it instead of anm_BR ***

if DisplayProgress
    fprintf('Finished generating binaural signals\n');
end

%% Transform binaural signal to time domain and listen
if headRotation
    % choose a single rotation index for listening
    rot_idx = 1;
    
    % create the binaural signal from selected rotation index
    bin_sig_t = [squeeze(bin_sig_rot_t(:, 1, rot_idx)),...
        squeeze(bin_sig_rot_t(:, 2, rot_idx))];
else      
    bin_sig_t = bin_sig_rot_t;    
end
% trim to size before power of 2 padding
bin_sig_t(size(anm_t, 1) + 1:end,:) = [];

if DisplayProgress
    fprintf('\n');
    disp('Binaural signals dimensions:');
    disp('===========================');
    disp(['bin_sig_t is of size (samples, ears (1 = left) ) = (' num2str(size(bin_sig_t, 1),'%d') ', ' num2str(size(bin_sig_t, 2),'%d') ')']);
end

% Listen to results - use headphones
soundsc(bin_sig_t, fs);










