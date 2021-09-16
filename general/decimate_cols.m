function [x, fs] = decimate_cols(x, fs, minFsNew)

% Author: Tom Shlomo, ACLab BGU, 2020

decimationFactor = floor(fs/minFsNew);
xbu = x;
for i=1:size(x,2)
    y = decimate(xbu(:,i), decimationFactor);
    if i==1
        x = zeros(size(y,1), size(xbu,2));
    end
    x(:,i) = y;
end
fs = fs/decimationFactor;

end