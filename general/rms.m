function v = rms(x, varargin)

% Author: Tom Shlomo, ACLab BGU, 2020

v = sqrt(mean(x.^2, varargin{:}));
end

