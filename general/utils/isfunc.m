function [ bool ] = isfunc( var )

% Author: Tom Shlomo, ACLab BGU, 2020

bool = isa(var, 'function_handle');

end

