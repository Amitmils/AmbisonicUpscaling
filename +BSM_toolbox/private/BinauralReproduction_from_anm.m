function bin_sig_rot_t = BinauralReproduction_from_anm(anm_t, HRTFpath, fs, N_BR, headRotation, rotAngles, WignerDpath, DisplayProgress)
arguments
    anm_t (:, :) double
    HRTFpath (1,1) string
    fs (1, 1) double
    N_BR (1, 1) double    
    headRotation (1, 1) logical = false
    rotAngles (1, :) double = 0
    WignerDpath (1,1) string = ''
    DisplayProgress (1, 1) logical = false   
end
% This function generates binaural signals from anm (Ambisonics format)
% Head-rotation is supported
%%Inputs:
% anm_t             : (time x (N+1)^2) anm signal to be reproduced
% HRTFpath          : (char) path to HRTF to be used
% fs                : (scalar) sampling frequency of signal/anm
% N_BR              : (scalar) maximal SH order to be reproduced
% headRotation      : (bool) true: generate rotated version of anm over azimuth - useful for head-tracking applications
% rotAngles         : (1 x rotation_angles) vector of rotation angles in radians
% WignerDpath       : (char) path to Wigner-D matrix
%%Outputs:
% bin_sig_rot_t     : (time x 2 x rot_angles) Binaural signal for each rotation angle (if there are rotations...)

%% ================= transform anm_t to frequency domain
NFFT = 2^nextpow2(size(anm_t, 1));
anm_f = fft(anm_t, NFFT, 1); 
% remove negative frequencies
anm_f = anm_f(1:NFFT/2+1, :);

%% ================= Generate binaural signals - Ambisonics format
if DisplayProgress
    fprintf('\n');
    disp('Binaural reproduction calculations:');
    disp('==================================');
end

anm_BR = anm_f.'; 
anm_BR = [anm_BR, conj(anm_BR(:, end-1:-1:2))];  % just to be consistent size-wise

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
bin_sig_rot_t = anm_HRTF_2_binaural(hobj, anm_BR(1:(N_BR+1)^2, :), N_BR, headRotation, rotAngles, WignerDpath);
% *** NOTE: it is much more efficient to use RIR-anm instead of signals containing
% anm, but this is an example for binaural reproduction from estimated anm.
% If RIR is given, use it instead of anm_BR ***

% trim to size before power of 2 padding
bin_sig_rot_t(size(anm_t,1)+1:end, :, :) = [];

if DisplayProgress
    fprintf('Finished generating binaural signals\n');
end

