function h = xlines(x,varargin)

% Author: Tom Shlomo, ACLab BGU, 2020

x = x(~isnan(x));
if isempty(x)
    h = [];
    return
end
for i=1:numel(x)
    h(i) = xline(x(i), varargin{:});
    if i>1
        hasbehavior(h(i), 'legend', false);   % line will not be in legend
    end
end

end

