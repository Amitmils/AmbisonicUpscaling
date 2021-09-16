function [H_l_nm_MagLS, H_r_nm_MagLS] = computeMagLS(H, f_vec, N, Y_N, cutOffFreq, Y_high)
% function computeMagLS computes the MagLS representation of the HRTFs
%
% Implementation based on Ambisonics book (https://link.springer.com/book/10.1007/978-3-030-17207-7)
%   Eqs.(4.57)-(4.59), for N<3 until (4.63)
%
% Inputs:
%       H          : HRTF in space/frequncy domain [# of directions X # of frequency bins X 2 (left/right)]
%       f_vec      : Frequencies vector [Hz]
%       N          : SH order
%       Y_N          : Complex normalized SH functions matrix of order N [(N+1)^2 X # of directions ]
%       cutOffFreq : Frequency cut-off for MagLS [Hz]
%       Y_high     : Complex normalized SH functions matrix of high order M>>N (used for Diffuse-Field Covariance Constraint, when N<3)
% Output:
%       H_l_nm_MagLS : SH coefficient of left HRTF [# of frequency bins X (N+1)^2]
%       H_r_nm_MagLS : SH coefficient of right HRTF [# of frequency bins X (N+1)^2]
%
% Written by Zamir Ben-Hur
% Nov 2020
%

%% initials
if nargin < 6 % don't use Diffuse-Field Covariance Constraint
    Y_high = [];
end

% find cut-off freq index
srcharray = abs(f_vec-cutOffFreq);
cutOffFreqInd = find(srcharray==min(srcharray));

% pseudo-inverse of matrix Y
pY = pinv(Y_N); 

H_l = squeeze(H(:,:,1)).'; % left HRTF
H_r = squeeze(H(:,:,2)).'; % right HRTF

H_l_nm_MagLS = zeros(size(H,2),(N+1)^2);
H_r_nm_MagLS = zeros(size(H,2),(N+1)^2);

%% Compute MagLS

% for freq < cutOffFreq, standard SH transform (Eq. 4.57)
H_l_nm_MagLS(1:cutOffFreqInd-1,:) = H_l(1:cutOffFreqInd-1,:)*pY; % SH transform - left ear
H_r_nm_MagLS(1:cutOffFreqInd-1,:) = H_r(1:cutOffFreqInd-1,:)*pY; % SH transform - right ear


% for freq >= cutOffFreq, MagLS (Eqs. 4.58, 4.59)
for freqInd = cutOffFreqInd : size(H,2)
    % left ear
    phi_est_l = angle(Y_N.'*H_l_nm_MagLS(freqInd-1,:).'); % Eq. (4.58)
    H_l_nm_MagLS(freqInd,:) = (abs(H_l(freqInd,:)) .* exp(1i*phi_est_l.')) * pY; %  Eq. (4.59)
    
    % right ear
    phi_est_r = angle(Y_N.'*H_r_nm_MagLS(freqInd-1,:).'); % Eq. (4.58)
    H_r_nm_MagLS(freqInd,:) = (abs(H_r(freqInd,:)) .* exp(1i*phi_est_r.')) * pY; %  Eq. (4.59)
end


% for N<3 use Diffuse-Field Covariance Constraint 
if N < 3 && ~isempty(Y_high)
    pY_high = pinv(Y_high); % pseudo-inverse of matrix Y_high
    H_l_nm_high = H_l*pY_high; % SH transform
    H_r_nm_high = H_r*pY_high; % SH transform
    
    for freqInd = 1 : size(H,2)
        H_SH_high = [H_l_nm_high(freqInd,:); H_r_nm_high(freqInd,:)].';
        H_SH = [H_l_nm_MagLS(freqInd,:); H_r_nm_MagLS(freqInd,:)].';
        
        % Eq. (4.63)
        X = chol(H_SH_high' * H_SH_high);
        X_est = chol(H_SH' * H_SH);
        
        [U,~,V] = svd(X_est'*X);
        
        H_SH_corr = [H_SH * inv(X_est) * V * U' * X].';
        
        H_l_nm_MagLS(freqInd,:) = H_SH_corr(1,:);
        H_r_nm_MagLS(freqInd,:) = H_SH_corr(2,:);
    end
    
end

end