function [ H ] = wbar( i, total, H )

% Author: Tom Shlomo, ACLab BGU, 2020

try %#ok<TRYNC>
    if nargin==0
        H.h=waitbar(0,'0');
        H.h.Children.Title.Interpreter = 'none';
        H.sTime = now();
    else
        if ischar(i) && strcmp(i, 'dros')
            close_all_waitbars(false);
            H = wbar();
            return
        end
        if i==total
            close(H.h);
            return
        end
        H.h = waitbar(i/total, H.h, ...
            [num2str(round(100*i/total)) '% Rem Time: '...
            datestr((now()-H.sTime)*(total-i)/i ,'HH:MM:SS')]);
    end
end

end