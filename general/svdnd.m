function [U, S, V] = svdnd(A, dims, r, wbarFlag)
% calculates svd of "slices" of a multidimensional array.
% A: is the multidimensional array.
% dims: a 2-vector contaning the dimensions of the slices.

% Author: Tom Shlomo, ACLab BGU, 2020

assert(length(dims)==2);
N = size(A,dims(1));
M = size(A,dims(2));

if nargin<3 || isempty(r)
    r = min(N,M);
end
if nargin<4 || isempty(wbarFlag)
    wbarFlag = size(A,3)>100;
end
permuteVec = [dims setdiff(1:ndims(A), dims)];
A = permute(A, permuteVec);

original_size = size(A);
A = reshape(A, N, M, []);
layers = size(A, 3);
U = zeros(N, r, layers);
if nargout>=2
    S = zeros(r, 1,layers);
end
if nargout>=3
    V = zeros(M, r, layers);
end
if wbarFlag
    H = wbar();
end
for l=1:layers
    [Ut, St, Vt] = svd(A(:,:,l), 'econ');
%     [Ut, St, Vt] = svds(A(:,:,l), r);
    U(:,:,l) = Ut(:, 1:r);
    if nargout>=2
        S(:,1,l) = diag(St(1:r,1:r));
    end
    if nargout>=3
        V(:,:,l) = Vt(:, 1:r);
    end
    if wbarFlag && ( mod(l,100)==0 || l==layers )
        wbar(l, layers, H);
    end
end
U = reshape(U, [N , r, original_size(3:end)]);
U = ipermute(U, permuteVec);

if nargout>=2
    V = reshape(V, [M , r, original_size(3:end)]);
    V = ipermute(V, permuteVec);
end
if nargout>=3
    S = reshape(S, [r , 1, original_size(3:end)]);
    S = ipermute(S, permuteVec);
end
end