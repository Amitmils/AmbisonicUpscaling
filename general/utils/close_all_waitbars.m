function [  ] = close_all_waitbars( displayFlag )

% Author: Tom Shlomo, ACLab BGU, 2020

if nargin==0
    displayFlag=true;
end
h = findall(0,'tag', 'TMWWaitbar');
close(h);
if displayFlag
    disp([num2str(length(h)) ' waitbars were closed']);
end
end
