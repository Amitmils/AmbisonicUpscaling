function h = scatter(th, ph, s, c, marker, varargin)
% Author: Tom Shlomo, ACLab BGU, 2020


if nargin<3
    s = 500;
end
if nargin<5
    marker = '.';
end
if isempty(ph) && size(th,2)==2
    ph = th(:,2);
    th = th(:,1);
end
[x,y] = hammer.project(th, ph);
h = scatter(x, y, s, c, marker, varargin{:});
hammer.axes(h.Parent);

end

