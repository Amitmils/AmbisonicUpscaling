% This script is an example for ACLtoolbox 
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

% Modified by:      Or Berebi
% Modified:         Aug 1 2021


function [bin_sig_t, fs] = Binuaural_reproduction_bilateral_ambisonics(anm_l_f,anm_r_f, fs,HRTFpath,N_PW)

N_BR = N_PW;                % SH order of Ambisonics signal

%=============== Choose which anm to use for binaural reproduction: simulated or estimated from array
anm_l_BR = anm_l_f; 
anm_l_BR = [anm_l_BR, conj(anm_l_BR(:, end-1:-1:2))];  % just to be consistent size-wise
anm_r_BR = anm_r_f; 
anm_r_BR = [anm_r_BR, conj(anm_r_BR(:, end-1:-1:2))];  % just to be consistent size-wise

clear anm_l_f anm_r_f anm_l_t anm_r_t 
% load HRTF to an hobj struct -
load(HRTFpath);                     % hobj is HRIR earo object - domains are given in hobj.dataDomain

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



[bin_sig_rot_t] = Bilateral_BinSigGen_HeadRotation_1RotIdx_ACL(hobj, anm_l_BR,anm_r_BR, N_BR);
clear anm_l_BR anm_r_BR

bin_sig_t = bin_sig_rot_t;
clear bin_sig_rot_t

end

