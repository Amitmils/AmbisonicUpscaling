function x = TikhonovReg(A, b, lambda, opt)
%% This function calculates the regularized LS: x = argmin_x{ || |A'x| - b ||^2 + lambda*|| x ||^2}
%%INPUTS:
% A - m by n matrix
% b - m size vector to be approximated
% lambda - regularization scalar
% tol - tolerance for iterative algorithm

%%OUTPUTS:
% x - n size vector of solutions

% get sizes
m = size(A, 1);
n = size(A, 2);

% Original Tikhonov regularization
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
end


end