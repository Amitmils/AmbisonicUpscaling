function x = TikhonovReg_MagLS_CVX(A, b, lambda, opt)
%% This function calculates the two-way partitioning problem of MagLS using CVX: x = argmin_x{ || |A'x| - b ||^2 + lambda*|| x ||^2}
%%INPUTS:
% A - m by n matrix
% b - m size vector to be approximated
% lambda - regularization scalar
% tol - tolerance for iterative algorithm

%%OUTPUTS:
% x - n size vector of solutions

%%References
% [1] Kassakian, P. (2005, March). Magnitude least-squares fitting via semidefinite programming with applications to beamforming and multidimensional filter design. In Proceedings.(ICASSP'05). IEEE International Conference on Acoustics, Speech, and Signal Processing, 2005. (Vol. 3, pp. iii-53). IEEE.

% check number of input arguments
switch nargin
    case 3
        % default
        opt = 1;
end
if nargin > 4
    error('Too many input arguments to TikhonovReg_MagLS_CVX.m function');
end

% Path to CVX solver
addpath(genpath('/Volumes/GoogleDrive/My Drive/Lior/Acoustics lab/Matlab/Research/Solvers/CVX/'));

% get sizes
m = size(A, 1);
n = size(A, 2);

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

% MagLS as real-valued partitioning problem
[x, err] = MLS_w_CVX(A, b, inv_mat, lambda);

end


function [x_opt, err] = MLS_w_CVX(A, b, inv_mat, lambda)
% Auxilary function to solve MLS with CVX as "real-valued partitioning problem"

m = size(A', 1);

% definitions according to [1] eqs. (2), (4), (6)
U = A' * inv_mat * A - eye(m);
B = diag(b);
W = (U * B)' * (U * B);
W_tilde = [real(W), -imag(W);...
           imag(W), real(W)];

cvx_begin sdp
variable C(2*m, 2*m) semidefinite
diag(C(1:m, 1:m)) + diag(C(m+1:2*m, m+1:2*m)) == ones(m, 1) 
% C >= 0
minimize(trace(C * W_tilde))
cvx_end

% feasile solution according to [1] page 2 "5. FEASIBLE SOLUTION" - second
% paragraph
[U, ~, ~] = svd(C);
s_opt = U(1:m, 1) + 1j * U(m+1:2*m, 1);
s_opt = s_opt ./ abs(s_opt);
x_opt = inv_mat * A * B * s_opt;
err = norm(abs(A' * x_opt) - abs(b), 2)^2 + lambda * norm(x_opt, 2)^2; 

end

