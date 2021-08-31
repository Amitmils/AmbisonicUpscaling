function e_mat = build_e_mat_positive(micPos,micPos_r,th_grid,ph_grid,kr)
    th_s = th_grid;
    ph_s = ph_grid;
    
    [th_e,ph_e,~] = c2s(micPos(1),micPos(2),micPos(3));
    [th_e_r,ph_e_r,~] = c2s(micPos_r(1),micPos_r(2),micPos_r(3));

    COS_OMEGA_0 = cos(th_s)*cos(th_e)+cos(ph_s - ph_e).*sin(th_s)*sin(th_e); %phase correction
    COS_OMEGA_1 = cos(th_s)*cos(th_e_r)+cos(ph_s - ph_e_r).*sin(th_s)*sin(th_e_r); %phase correction

    d1          = (COS_OMEGA_0 - COS_OMEGA_1);
    e_mat       = exp(1i*d1.'*kr);

end
