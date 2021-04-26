function h = text(th, ph, z, txt, varargin)
if isempty(th)
    h = [];
    return
end
% Author: Tom Shlomo, ACLab BGU, 2020

if isempty(ph) && size(th,2)==2
    ph = th(:,2);
    th = th(:,1);
end
if isempty(z)
    z = zeros(size(th));
end
[x,y] = hammer.project(th, ph);
h = text(x, y, z, txt, varargin{:});
hammer.axes(h(1).Parent);

end