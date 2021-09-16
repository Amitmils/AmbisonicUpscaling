function x = TikhonovReg_MagLS(A, b, lambda, opt, tol)
%% This function calculates the iterative solution to the regularized magnitude LS: x = argmin_x{ || |Ax| - b ||^2 + lambda*|| x ||^2}
%%INPUTS:
% A - m by n matrix
% b - m size vector to be approximated
% lambda - regularization scalar
% tol - tolerance for iterative algorithm

%%OUTPUTS:
% x - n size vector of solutions

%%References
% [1] Setsompop, K., et al. "Magnitude least squares optimization for parallel radio frequency excitation design demonstrated at 7 Tesla with eight channels." Magnetic Resonance in Medicine: An Official Journal of the International Society for Magnetic Resonance in Medicine 59.4 (2008): 908-915.

% get sizes
m = size(A, 1);
n = size(A, 2);

% initialize 
err_prev = inf;

% first iteration -  Original Tikhonov regularization
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
inv_mat = (U * diag(Sig) * V');

switch opt
    case 1
        x = inv_mat * ((1 / lambda) * A * conj(b));
    case 2
        x = inv_mat * (A * conj(b));     
%         err =  norm(A' * x - conj(b), 2)^2 + lambda * norm(x, 2)^2; 
end
err =  norm(A' * x - conj(b), 2)^2 + lambda * norm(x, 2)^2; 

iter = 1;
while (err_prev - err) > tol
    err_prev = err;
    phase = angle(A' * x);
    zi = exp(1j * phase);
    [x, err] = LS_w_mag_phase(A, b, lambda, inv_mat, opt, zi);
    
    iter = iter + 1;
end

fprintf('Finished Mag-LS in %d iterations\n', iter);


end



function [x, err] = LS_w_mag_phase(A, b, lambda, inv_mat, opt, zi)
% Auxilary function to calculate the magnitude LS part

switch opt
    case 1
        x = inv_mat * ((1 / lambda) * (A * (abs(b) .* zi)));
    case 2
        x = inv_mat * (A * (abs(b) .* zi));     
end
err =  norm(A' * x - abs(b) .* zi, 2)^2 + lambda * norm(x, 2)^2; 
end
