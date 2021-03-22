function c = soundspeed(temp_celcius)
% return the speed of sound in meters/sec.
% temp_celcius (optional): the air tempature in celcius.
%
% Author: Tom Shlomo, ACLab BGU, 2020


if nargin==0 || isempty(temp_celcius)
    c = 343; % m/sec
    return
end
c = 20.05 * sqrt( temp_celcius + 273.15 );

end