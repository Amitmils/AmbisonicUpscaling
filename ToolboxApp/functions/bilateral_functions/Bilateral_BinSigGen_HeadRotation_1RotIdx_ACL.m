function [bin_sig_rot_t] = Bilateral_BinSigGen_HeadRotation_1RotIdx_ACL(hobj, anm_l_f,anm_r_f, N)
arguments
    hobj earo
    anm_l_f (:, :) double
    anm_r_f (:, :) double
    N (1, 1) double
   
end
% This function generate BRIR with head rotation for a given HRTF (hobj)
% and plane-waves anm
%
% Zamir Ben-Hur
% 4.3.2015
% Modified (01.08.2021 - Or Berebi)

% [3] Rafaely, Boaz, and Amir Avni. "Interaural cross correlation in a sound field represented by spherical harmonics." The Journal of the Acoustical Society of America 127.2 (2010): 823-828.

hobj = HRTF_phaseCorrection(hobj, 0); %Zamir's ear-alignd HRTF function

% SH transform
if strcmp(hobj.dataDomain{2},'SPACE')
    hobj = hobj.toSH(N, 'SRC');
else
    warning('hobj is already in the SH domain')
end   

% Transform HRTFs to frequency domain
NFFT = size(anm_l_f, 2);
if strcmp(hobj.dataDomain{1},'FREQ') && size(hobj.data,2)~=ceil(NFFT/2)+1
    hobj=hobj.toTime(); 
end
hobj = hobj.toFreq(NFFT);
% Trim negative frequencies
hobj.data = hobj.data(:, 1:NFFT/2 + 1, :);
Hnm_lt = hobj.data(:, :, 1);
Hnm_rt = hobj.data(:, :, 2);

anm_l_f = anm_l_f(:, 1:NFFT/2 + 1);
anm_r_f = anm_r_f(:, 1:NFFT/2 + 1);


anm_tilde_l = tildize(N) * anm_l_f;
anm_tilde_r = tildize(N) * anm_r_f;


bin_sig_rot_t = zeros(NFFT, 2);

% Rotation matrix
%D = eye((N + 1)^2);
% Hnm_lt_rot = (hobj.data(:, :, 1).' * D).';
% Hnm_rt_rot = (hobj.data(:, :, 2).' * D).';





% Generate BRIR    
% Ambisonics format binaural reproduction - see [3] eq. (9)
pl_f = sum(anm_tilde_l .* Hnm_lt, 1).';
pr_f = sum(anm_tilde_r .* Hnm_rt, 1).';

plr_f = [pl_f, pr_f];
% pad negative frequencies with zeros (has no effect since we use ifft with "symmetric" flag)
plr_f(end+1:NFFT, :) = 0;
bin_sig_rot_t(:, :) = ifft(plr_f, [], 1, 'symmetric');       


end

% Internal functions
function [ Perm ] = tildize( N )
%A_TILD Summary of this function goes here
%   Detailed explanation goes here
Perm=(-1).^(2:(N+1)^2+1);
Perm=diag(Perm);
for n=0:N
    Perm(n^2+1:n^2+2*n+1,n^2+1:n^2+2*n+1)=fliplr(Perm(n^2+1:n^2+2*n+1,n^2+1:n^2+2*n+1));
end
end
