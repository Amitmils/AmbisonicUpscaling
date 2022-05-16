function hobj_rot = RotateHRTFwigner(hobj_freq_grid, N_HRTF_rot, head_rot)
%%hobj_rot_BSM.m
% Rotate HRTFs azimuth grid according to head rotation
% This is for the compensation part of BSM
%%Inputs:
% hobj_freq_grid        : (earo format) HRTF object in SPACE-FREQ domain
% N_HRTF_rot            : (scalar) Maximal SH order of rotated HRTF
% head_rot              : (1 x 2) vector of rotation degrees (theta, phi)
%%Outputs:  
% hobj_rot_BSM          : (earo format) Rotated HRTF in SH-FREQ domain
    
    %% ================= Rotate HRTFs according to head rotation - new        
    hobj_rot = hobj_freq_grid;
    
    % Transform HRTF to SH
    if strcmp(hobj_rot.dataDomain{2},'SPACE')
        hobj_rot = hobj_rot.toSH(N_HRTF_rot,'SRC');
    else
        warning('hobj_rot is already in the SH domain')
    end
    
    % Wigner-D rotation matrix
    gamma = 0;                            % first counter-clockwise rotation about z-axis
    beta = head_rot(:, 1);  % second counter-clockwise rotation about y-axis
    alpha = head_rot(:, 2);               % third counter-clockwise rotation about z-axis
    D_wig = WignerDM(N_HRTF_rot, alpha, beta, gamma);
    
    % Rotate HRTFs
    hobj_rot.data(:, :, 1)=(hobj_rot.data(:, :, 1).' * D_wig).';
    hobj_rot.data(:, :, 2)=(hobj_rot.data(:, :, 2).' * D_wig).';    


end