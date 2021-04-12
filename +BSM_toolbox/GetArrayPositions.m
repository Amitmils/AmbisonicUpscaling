function [th_array, ph_array, ph_rot_array] = GetArrayPositions(arrayType, n_mic, array_rot_az)
%% GetArrayPositions.m
% Calculate array positions (spherical coordinates) given arrayType
%%Inputs:
% arrayType           : (scalar) 0 - spherical array, 1 - semi-circular array, 2 - full-circular array
% n_mic               : (scalar) number of mics
% array_rot_az        : (scalar) degree of azimuth rotation (rad)
%%Outputs:
% th_array            : (1 x n_mic) Microphone elevation
% ph_array            : (1 x n_mic) Original Microphone azimuth
% ph_rot_array        : (1 x n_mic) Rotated Microphone azimuth

    %ArrayPos = zeros(3, n_mic);
    if arrayType == 0
        %================= generate coordinates of spherical open array        
        [~, th_array, ph_array] = SpiralSampleSphere_w_weights(n_mic);        
        th_array = th_array.';
        ph_array = ph_array.'; 

        % rotate array in azimuth
        ph_rot_array = wrapTo2Pi(ph_array + array_rot_az); 

    elseif arrayType == 1
        % parameters of semi circular array
        mic_idx = 1:n_mic;
        ph_array = deg2rad(90 - 180 / (n_mic - 1) * (mic_idx - 1));
        ph_array = wrapTo2Pi(ph_array);
        th_array = deg2rad(90) * ones(size(ph_array));

        % rotate array in azimuth
        ph_rot_array = wrapTo2Pi(ph_array + array_rot_az);             

    elseif arrayType == 2
        % parameters of fully circular array
        mic_idx = 1:n_mic;
        ph_array = deg2rad(360 / (n_mic) * (mic_idx - 1));
        ph_array = wrapTo2Pi(ph_array);
        th_array = deg2rad(90) * ones(size(ph_array));        

        % rotate array in azimuth
        ph_rot_array = wrapTo2Pi(ph_array + array_rot_az); 

    end 
end



