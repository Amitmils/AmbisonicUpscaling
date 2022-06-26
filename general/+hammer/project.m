function [x, y] = project(th, ph)
% https://en.wikipedia.org/wiki/Hammer_projection
%
% Author: Tom Shlomo, ACLab BGU, 2020

if nargin==1
    assert( size(th,2) == 2 );
    ph = th(:,2);
    th = th(:,1);
else
    th = th + zeros(size(ph));
    ph = ph + zeros(size(th));
end

%% wrap th to [0,pi] and ph to [-pi,pi]
th = mod2(th, 2*pi);
I = th<0;
th(I) = -th(I);
ph(I) = ph(I) + pi;
ph = mod2(ph, 2*pi);

%% get lambda and phi
lambda2 = ph/2;
phi = pi/2-th;

cos_phi = cos(phi);
denom = sqrt(1 + cos_phi .* cos(lambda2));
x = sqrt(8) .* cos_phi .* sin(lambda2) ./ denom;
y = sqrt(2) .* sin(phi) ./ denom;

end

