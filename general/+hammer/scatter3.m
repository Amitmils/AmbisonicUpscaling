function h = scatter3(th, ph, z, s, c, marker, varargin)
% Author: Tom Shlomo, ACLab BGU, 2020


if nargin<4
    s = 500;
end
if nargin<6
    marker = '.';
end
if isempty(ph) && size(th,2)==2
    ph = th(:,2);
    th = th(:,1);
end
[x,y] = hammer.project(th, ph);
h = scatter3(x, y, z, s, c, marker, varargin{:});
hammer.axes(h.Parent);

end