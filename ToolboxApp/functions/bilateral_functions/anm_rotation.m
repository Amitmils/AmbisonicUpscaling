function anm_k_D_c = anm_rotation(anm_k,micPos,micPos_r,D,Y,Yp,th_grid,ph_grid,kr)

    anm_k_D = D*anm_k;              %SH/Freq (wigner D)
    a_k_D = Y.'*anm_k_D;            %Space/Freq

    e_mat     = build_e_mat_positive(micPos,micPos_r,th_grid,ph_grid,kr);

    a_k_D_c   = a_k_D.*e_mat;         %Space/Freq (phase correction)

    anm_k_D_c = (a_k_D_c.'*Yp).';   %SH/Freq



end