function [bin_sig_rot_t, rotAngles] = BinSigGen_HeadRotation_ACL(hobj, anm_f, N, headRotation, WignerDpath)
arguments
    hobj earo
    anm_f (:, :) double
    N (1, 1) double
    headRotation (1, 1) logical = false
    WignerDpath (1,1) string = ''
end
% This function generate BRIR with head rotation for a given HRTF (hobj)
% and plane-waves anm
%
% Zamir Ben-Hur
% 4.3.2015
% Modified (04.02.2021 - Lior Madmoni)

% [3] Rafaely, Boaz, and Amir Avni. "Interaural cross correlation in a sound field represented by spherical harmonics." The Journal of the Acoustical Society of America 127.2 (2010): 823-828.

if headRotation
    % load WignerD Matrix
    D = load(WignerDpath);
    D = D.D;
    DN = (N + 1)^2; % size of the wignerD matrix
    D_allAngles = D(:, 1 : DN);
    
    % choose rotation angles
    rotAngles = deg2rad(0:1:359);
else
    % no head rotation    
    rotAngles = deg2rad(0);
end

% SH transform
if strcmp(hobj.dataDomain{2},'SPACE')
    hobj = hobj.toSH(N, 'SRC');
else
    warning('hobj is already in the SH domain')
end   

% Transform HRTFs to frequency domain
NFFT = size(anm_f, 2);
if strcmp(hobj.dataDomain{1},'FREQ') && size(hobj.data,2)~=ceil(NFFT/2)+1, hobj=hobj.toTime(); end;
hobj = hobj.toFreq(NFFT);

% Trim negative frequencies
hobj.data = hobj.data(:, 1:NFFT/2 + 1, :);
anm_f = anm_f(:, 1:NFFT/2 + 1);

% Iterate through head rotation angles
if ~hobj.shutUp, fprintf('Generating BRIRs...\n'); end;
anm_tilde = tildize(N) * anm_f;

if ~hobj.shutUp
    fprintf('   --->Head rotation of azimuth = %3d',round(rad2deg(rotAngles(1))));
end

bin_sig_rot_t = zeros(NFFT, 2, length(rotAngles));
for angle_idx = 1:length(rotAngles)        
    
    if ~hobj.shutUp && angle_idx ~=1
        fprintf('\b\b\b%3d',round(rad2deg(rotAngles(angle_idx))));
    end
    
    % Rotation matrix
    if headRotation
        rot_idx = round(rad2deg(rotAngles(angle_idx)));
        if rot_idx == 0
            rot_idx = 1;
        elseif rot_idx == 360
            rot_idx = 359;
        end    
        D = diag(D_allAngles(rot_idx, :));
    else
        D = eye((N + 1)^2);
    end
    
    Hnm_lt_rot = (hobj.data(:, :, 1).' * D).';
    Hnm_rt_rot = (hobj.data(:, :, 2).' * D).';
    
    % Generate BRIR    
    % Ambisonics format binaural reproduction - see [3] eq. (9)
    pl_f = sum(anm_tilde .* Hnm_lt_rot, 1).';
    pr_f = sum(anm_tilde .* Hnm_rt_rot, 1).';
    
    plr_f = [pl_f, pr_f];
    % pad negative frequencies with zeros (has no effect since we use ifft with "symmetric" flag)
    plr_f(end+1:NFFT, :) = 0;
    bin_sig_rot_t(:, :, angle_idx) = ifft(plr_f, [], 1, 'symmetric');       
end

if ~hobj.shutUp
    fprintf('\n');
end

end

%% Internal functions
function [ Perm ] = tildize( N )
%A_TILD Summary of this function goes here
%   Detailed explanation goes here
Perm=(-1).^(2:(N+1)^2+1);
Perm=diag(Perm);
for n=0:N
    Perm(n^2+1:n^2+2*n+1,n^2+1:n^2+2*n+1)=fliplr(Perm(n^2+1:n^2+2*n+1,n^2+1:n^2+2*n+1));
end
end
