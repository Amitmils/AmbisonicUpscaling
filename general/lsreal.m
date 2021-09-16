function x = lsreal(A,b)
% Solves the following optimization problem, with variable x:
% minmize || A*x - b ||_2
% under the constraint that that all entries of x are real.
% (A and b can and should be complex here, otherwise the solution is given simply by A\b)
%
% Author: Tom Shlomo, ACLab BGU, 2020

x = [real(A); imag(A)]\[real(b); imag(b)];

end