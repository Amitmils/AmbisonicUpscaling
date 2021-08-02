function anm_k_D_c = anm_rotation_LPF(anm_k,micPos,micPos_r,D,Y,Yp,th_grid,ph_grid,kr,Fs,Fc,width)


    nfft = 2*(size(anm_k,2) - 1);

    anm_k_D = D*anm_k;              %SH/Freq (wigner D)
    a_k_D = Y.'*anm_k_D;            %Space/Freq

    e_mat     = build_e_mat_positive(micPos,micPos_r,th_grid,ph_grid,kr);

    a_k_D_c   = a_k_D.*e_mat;         %Space/Freq (phase correction)

    
    %mix translation and rotation
    %-------------------------------------
%     width = 3e3; %the fadout filter width
%     Fc = 3e3; %cutoff freq
    n=double(linspace(0,Fs/2,nfft/2+1)); % frequency range
    
    [~,ind1] = min(abs(n - Fc));
    [~,ind2] = min(abs(n - (Fc+width)));
    

    filter_out              = zeros(size(n));
    filter_out(1:ind1)      = 1;
    filter_out(ind1:ind2-1) = linspace(1,0,(ind2-ind1));
    filter_in               = ones(size(n));
    filter_in               = filter_in - filter_out;
    filter_out              = repmat(filter_out,[size(a_k_D_c,1),1]);
    filter_in               = repmat(filter_in,[size(a_k_D_c,1),1]);

    a_k_D_c(:,1:size(filter_out,2)) = a_k_D_c(:,1:size(filter_out,2)).*filter_out;  %fade out the translation
    a_k_D(:,1:size(filter_out,2))   = a_k_D(:,1:size(filter_out,2)).*filter_in;       % fade in the rotation
    
    a_k_LPF     = a_k_D_c(:,1:size(filter_out,2))+a_k_D(:,1:size(filter_out,2));        %mix both
    %a_k_LPF     = a_k_LPF.';
    %a_k_LPF     = [a_k_LPF;flipud(conj(a_k_LPF(2:end-1,:)))].';                         %calc negative freq
    anm_k_D_c = Yp.' * a_k_LPF;
    %anm_k_D_c   = (a_k_LPF.'*Yp).';   %SH/Freq

    
%     a_k_D_spectra = (a_k_LPF.'*Yp).';
%     SH_spectra_v5(a_k_D_spectra,Fs,0.008,new_N)
%     e_mat_spectra = (e_mat.'*Yp).';


end
