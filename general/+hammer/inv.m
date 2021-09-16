function [th, ph] = inv(x, y)
% Author: Tom Shlomo, ACLab BGU, 2020


z = sqrt(1 - (x/4).^2 - (y./2).^2);
lambda = 2*atan2(z.*x, 2*(2*z.^2-1));
phi = asin(z.*y);

ph = lambda;
th = pi/2-phi;
end