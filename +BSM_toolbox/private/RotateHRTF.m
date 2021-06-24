function hobj_rot = RotateHRTF(hobj_freq_grid, N_HRTF_rot, D_allAngles, head_rot_az)
%%hobj_rot_BSM.m
% Rotate HRTFs azimuth grid according to head rotation
% This is for the compensation part of BSM
%%Inputs:
% hobj_freq_grid        : (earo format) HRTF object in SPACE-FREQ domain
% N_HRTF_rot            : (scalar) Maximal SH order of rotated HRTF
% D_allAngles           : ((N_HRTF_rot + N_HRTF_rot)^2 x (N + 1)^2) rotation matrix
% head_rot_az           : (scalar) current degree of rotation
%%Outputs:
% hobj_rot_BSM          : (earo format) Rotated HRTF in SH-FREQ domain
    
    %% ================= Rotate HRTFs according to head rotation - new        
    hobj_rot = hobj_freq_grid;
    
    % Transform HRTF to SH
    %N_HRTF_rot = sqrt(size(D_allAngles, 1)) - 1;
    if strcmp(hobj_rot.dataDomain{2},'SPACE')
        hobj_rot = hobj_rot.toSH(N_HRTF_rot,'SRC');
    else
        warning('hobj_rot is already in the SH domain')
    end
    
    % Rotation matrix
    rot_idx = round(rad2deg( head_rot_az ));
    if rot_idx == 0
        rot_idx = 1;
    elseif rot_idx == 360
        rot_idx = 359;
    end
    D_curr = diag(D_allAngles(rot_idx, :));

    hobj_rot.data(:, :, 1)=(hobj_rot.data(:, :, 1).' * D_curr).';
    hobj_rot.data(:, :, 2)=(hobj_rot.data(:, :, 2).' * D_curr).';    


end