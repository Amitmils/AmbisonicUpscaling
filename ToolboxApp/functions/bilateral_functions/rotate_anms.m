function [anm_l_k_A,anm_r_k_A] = rotate_anms(anmk_l,anmk_r,rot_ang,N,head_vec,kr,LPF,Fc,Width,fs)
    % make Wigner D
    % -------------------------------
    rot_ang = [0,0,rot_ang];
    D       = WignerDM(N,-rot_ang(1),-rot_ang(2),-rot_ang(3));
    % Rotate the head
    % -------------------------------
    [x0,y0,z0]  = s2c(head_vec(2),head_vec(3),head_vec(1)); %left ear vector in cartesians
    LeftEar     = [x0,y0,z0];
    inc = abs(LeftEar) > 1e-10;
    LeftEar = LeftEar.*inc;

    [x0,y0,z0]  = s2c(head_vec(4),head_vec(5),head_vec(1)); %right ear vector in cartesians
    RightEar    = [x0,y0,z0];
    inc = abs(RightEar) > 1e-10;
    RightEar = RightEar.*inc;

    M = getRotationMat_ZYZ(-rot_ang(1),-rot_ang(2),-rot_ang(3));
    LeftEar_r_M = (M*LeftEar.').';
    RightEar_r_M = (M*RightEar.').';


    %Rotate anm's
    %---------------------------------------
    [a_grid,th_grid,ph_grid] = equiangle_sampling(N+3);
    %Y = sh2(N,th_grid,ph_grid);
    Y = shmat(N,[th_grid.',ph_grid.'],true,true); % spherical harmonics matrix

    Yp=diag(a_grid)*Y';

    if LPF
        disp("Bilateral Ambisonics rotation with band-limited translation")
        anm_l_k_A    = anm_rotation_LPF(anmk_l,LeftEar,LeftEar_r_M,D,Y,Yp,th_grid,ph_grid,kr,fs,Fc,Width);
        anm_r_k_A    = anm_rotation_LPF(anmk_r,RightEar,RightEar_r_M,D,Y,Yp,th_grid,ph_grid,kr,fs,Fc,Width);
    else
        anm_l_k_A       = anm_rotation(anmk_l,LeftEar,LeftEar_r_M,D,Y,Yp,th_grid,ph_grid,kr);
        anm_r_k_A       = anm_rotation(anmk_r,RightEar,RightEar_r_M,D,Y,Yp,th_grid,ph_grid,kr);
    end

end

