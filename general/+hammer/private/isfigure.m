function [ b ] = isfigure( fig )
% Author: Tom Shlomo, ACLab BGU, 2020

b = false(size(fig));
for i=1:numel(fig)
    b(i) = isa(fig(i), 'matlab.ui.Figure');
end

end

