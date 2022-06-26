function [x, phase_last] = TikhonovReg_MagLS_v2(A, b, lambda, phase_init, opt, tol, max_iter_number)
%% This function calculates the iterative solution to the regularized magnitude LS: x = argmin_x{ || |A'x| - b ||^2 + lambda*|| x ||^2}
%%INPUTS:
% A - m by n matrix
% b - m size vector to be approximated
% lambda - regularization scalar
% tol - tolerance for iterative algorithm

%%OUTPUTS:
% x - n size vector of solutions

%%References
% [1] Setsompop, K., et al. "Magnitude least squares optimization for parallel radio frequency excitation design demonstrated at 7 Tesla with eight channels." Magnetic Resonance in Medicine: An Official Journal of the International Society for Magnetic Resonance in Medicine 59.4 (2008): 908-915.

% check number of input arguments
switch nargin
    case 3
        % default
        phase_init = pi / 2;    % according to [1]
        opt = 1;
        tol = 1e-10;
        max_iter_number = 1E5;
    case 4
        % default        
        opt = 1;
        tol = 1e-10;
        max_iter_number = 1E5;
    case 5        
        % default   
        tol = 1e-10;
        max_iter_number = 1E5;
    case 6
        % default   
        max_iter_number = 1E5;     
end
if nargin > 7
    error('Too many input arguments to TikhonovReg_MagLS_v2.m function');
end

% get sizes
m = size(A, 1);
n = size(A, 2);

% initialize - according to [1]
phase = phase_init;
zi = exp(1j * phase);
err_prev = inf;

% calcualte matrix inversion only once
switch opt
    case 1
        mat_to_inv = ((1 / lambda) * (A * A') + eye(m));
    case 2
        mat_to_inv = ((A * A') + lambda * eye(m));
end
% robust "inversion"
tol_inv = 1 + max(size(mat_to_inv))*eps(norm(mat_to_inv));  % the tolerance is the minimal singular value due to the noise (1) + increase in the singular value due to the norm of hte matrix to be inverted
[U, Sig, V] = svd(mat_to_inv);       
Sig = diag(Sig); 
Sig(Sig <= tol_inv) = 0;
Sig(Sig ~= 0) = 1 ./ Sig(Sig ~= 0);
inv_mat = (V * diag(Sig) * U');

% first iteration -  Original Tikhonov regularization
iter = 1;
[x, err] = LS_w_mag_phase(A, b, lambda, inv_mat, opt, zi);

while (err_prev - err) > tol && iter <= max_iter_number
    err_prev = err;
    phase = angle(A' * x);
    zi = exp(1j * phase);
    [x, err] = LS_w_mag_phase(A, b, lambda, inv_mat, opt, zi);
    
    iter = iter + 1;
end

phase_last = phase;

end



function [x, err] = LS_w_mag_phase(A, b, lambda, inv_mat, opt, zi)
% Auxilary function to calculate the magnitude LS part

switch opt
    case 1
        x = inv_mat * ((1 / lambda) * (A * (abs(b) .* zi)));
    case 2
        x = inv_mat * (A * (abs(b) .* zi));     
end

err =  norm(A' * x - (abs(b) .* zi), 2)^2 + lambda * norm(x, 2)^2; 

end

