function h = ylines(y,varargin)

% Author: Tom Shlomo, ACLab BGU, 2020

y = y(~isnan(y));
if isempty(y)
    h = [];
    return
end
for i=1:numel(y)
    h(i) = yline(y(i), varargin);
end

end

