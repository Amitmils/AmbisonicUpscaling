function h = plot3(th, ph, z, varargin)
% Author: Tom Shlomo, ACLab BGU, 2020


if size(th,2) == 2
    ph = th(:,2);
    th = th(:,1);
end
if isscalar(z)
    z = repmat(z, size(th,1),1);
end
[x,y] = hammer.project(th, ph);
h = plot3(x, y, z, varargin{:});
hammer.axes(h.Parent);

end