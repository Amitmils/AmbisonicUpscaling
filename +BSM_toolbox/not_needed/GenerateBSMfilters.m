function [c_BSM_l, c_BSM_r] = GenerateBSMfilters(BSMobj, hobj_freq_grid)
%% GenerateBSMfilters.m
% Generate the BSM filters - either using complex or magnitude LS
% optimization
%% Inputs:
% BSMobj            : (MATLAB object) containing parameters
% hobj_freq_grid    : HRTF object in earo format interpolated to desired frequencies
%% Outputs:
% c_BSM_l           : (n_mic x freq) BSM filters for left ear in freq. domain
% c_BSM_l           : (n_mic x freq) BSM filters for right ear in freq. domain

    %init
    r_array         = BSMobj.r_array;
    N_PW            = BSMobj.N_PW;
    freqs_sig       = BSMobj.freqs_sig;
    magLS           = BSMobj.magLS;
    f_cut_magLS     = BSMobj.f_cut_magLS;
    tol_magLS       = BSMobj.tol_magLS;
    max_iter_magLS  = BSMobj.max_iter_magLS;
    c               = BSMobj.c;
    n_mic           = BSMobj.n_mic;                
    th_array        = BSMobj.th_array;
    ph_array        = BSMobj.ph_array;
    th_BSMgrid_vec  = BSMobj.th_BSMgrid_vec;
    ph_BSMgrid_vec  = BSMobj.ph_BSMgrid_vec;
    normSV          = BSMobj.normSV;
    SNR_lin         = BSMobj.SNR_lin;
    inv_opt         = BSMobj.inv_opt;
    rigidArray      = BSMobj.rigidArray;    
    
    c_BSM_l     = zeros(n_mic, length(freqs_sig));
    c_BSM_r     = zeros(n_mic, length(freqs_sig));      

    % initialize phase for magLS
    if magLS
        phase_init_l_magLS = pi / 2;    % according to [1]
        phase_init_r_magLS = pi / 2;    % according to [1]        
    end
    
    % calculation of SV with SH: exp = Ymic * B * Y'
    N_SV = N_PW;
    Ymic   = sh2(N_SV, th_array, ph_array).';  % with copensation [mic x SH]  
    Y = conj(sh2(N_SV, th_BSMgrid_vec, ph_BSMgrid_vec));  % [SH x PWs]
    for f = 1:length(freqs_sig)        
        lambda = c / freqs_sig(f); % wave-length
        kr = 2 * pi / lambda * r_array;
        %N_SV = ceil(kr)+1;
        %N_SV = 30;        
        Bn_mat = BnMat(N_SV, kr, kr, rigidArray);                    
        V_k   = (Ymic * diag(Bn_mat) * Y);                     

        %normalize steering vectors
        if normSV
            V_k_norm = sqrt(sum(abs(V_k).^2, 1));
            V_k = V_k ./ V_k_norm;
        end

        %%================= solve for vector c, (V * c = h)
        h_l = hobj_freq_grid.data(:, f, 1);
        h_r = hobj_freq_grid.data(:, f, 2);

        if sum(sum(isnan(V_k)))
            c_BSM_l(:, f) = zeros(n_mic, 1);
            c_BSM_r(:, f) = zeros(n_mic, 1);
        else                
            % solution to Tykhonov regularization problem            
            if magLS && freqs_sig(f) >= f_cut_magLS                    
                % ===== MAG LS                                    
                [c_BSM_l(:, f), phase_last] = TikhonovReg_MagLS_v2(V_k, h_l, (1 / SNR_lin), phase_init_l_magLS, inv_opt, tol_magLS, max_iter_magLS);                                        
                phase_init_l_magLS = phase_last;
                [c_BSM_r(:, f), phase_last] = TikhonovReg_MagLS_v2(V_k, h_r, (1 / SNR_lin), phase_init_r_magLS, inv_opt, tol_magLS, max_iter_magLS);                                        
                phase_init_r_magLS = phase_last;
                % ==============

            else
                % ===== CMPLX LS                    
                c_BSM_l(:, f) = TikhonovReg(V_k, h_l, (1 / SNR_lin), inv_opt);
                c_BSM_r(:, f) = TikhonovReg(V_k, h_r, (1 / SNR_lin), inv_opt);
                % ==============
            end                                
        end            
    end  
    
    
end




