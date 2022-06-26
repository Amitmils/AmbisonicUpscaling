%% HRTF2BSMgrid.m
% Convert HRTFs from original grid to BSM grid of assumed sources
% The function uses the hobj (earo) object - hopefully to be deprecated in
% near future
%% Inputs:
% hobj                :  HRTF object in earo format
% N_HRTF              : (scalar) maximal SH order of the transformation
% th_BSMgrid_vec      :  (Q x 1) elevation angles 
% ph_BSMgrid_vec      :  (Q x 1) azimuth angles 
%% Outputs:
% hobj                :  HRTF object in earo format in new directions

function [hobj, th_BSMgrid_vec, ph_BSMgrid_vec] = HRTF2BSMgrid(hobj, N_HRTF, th_BSMgrid_vec, ph_BSMgrid_vec)
    % interpolate HRTF to new grid            
    hobj = hobj.toSH(N_HRTF, 'SRC');
    hobj = hobj.toSpace('SRC', th_BSMgrid_vec, ph_BSMgrid_vec);
end





