function [U, lambda] = eignd(A, dims)
% calculates eigendecomposition of "slices" of a multidimensional array.
% A: is the multidimensional array.
% dims: a 2-vector contaning the dimensions of the slices.
%
% Author: Tom Shlomo, ACLab BGU, 2020


assert(length(dims)==2);
assert(size(A,dims(1))==size(A,dims(2)));
Q = size(A,dims(1));


permuteVec = [dims setdiff(1:ndims(A), dims)];
A = permute(A, permuteVec);

original_size = size(A);
A = reshape(A, Q, Q, []);
U = zeros(size(A));
size_lambda = size(A);
size_lambda(2) = 1;
lambda = zeros(size_lambda);

wbarFlag = size(A,3)>100;
if wbarFlag
    H = wbar();
end
for i=1:size(A,3)
    [U(:,:,i), lambda(:,1,i)] = eig_sorted(A(:,:,i));
    if wbarFlag && (mod(i,100)==0 || i==size(A,3))
        wbar(i, size(A,3), H);
    end
end
U = reshape(U, original_size);
lambda = reshape(lambda, [Q 1 original_size(3:end)]);
U = ipermute(U, permuteVec);
lambda = ipermute(lambda, permuteVec);

end