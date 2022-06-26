function [c_BSM_l, c_BSM_r] = GenerateBSMfilters_faster(BSMobj, V_k, hobj_freq_grid)
%% GenerateBSMfilters.m
% Generate the BSM filters - either using complex or magnitude LS
% optimization
%% Inputs:
% BSMobj            : (MATLAB object) containing parameters
% V_k               : (n_mic x Q x freq]) steering vectors (ATF)
% hobj_freq_grid    : HRTF object in earo format interpolated to desired frequencies
%% Outputs:
% c_BSM_l           : (n_mic x freq) BSM filters for left ear in freq. domain
% c_BSM_l           : (n_mic x freq) BSM filters for right ear in freq. domain

    %init
    freqs_sig       = BSMobj.freqs_sig;
    magLS           = BSMobj.magLS;
    f_cut_magLS     = BSMobj.f_cut_magLS;
    tol_magLS       = BSMobj.tol_magLS;
    max_iter_magLS  = BSMobj.max_iter_magLS;    
    magLS_cvx       = BSMobj.magLS_cvx;
    n_mic           = BSMobj.n_mic;                    
    normSV          = BSMobj.normSV;
    SNR_lin         = BSMobj.SNR_lin;
    inv_opt         = BSMobj.inv_opt;    
    %
    c_BSM_l     = zeros(n_mic, length(freqs_sig));
    c_BSM_r     = zeros(n_mic, length(freqs_sig));      
    
    % initialize phase for magLS
    if magLS
        phase_init_l_magLS = pi / 2;    % according to [1]
        phase_init_r_magLS = pi / 2;    % according to [1]        
    end
    
    %normalize steering vectors    
    if normSV
        V_k_norm = vecnorm(V_k, 2, 1);
        V_k = V_k ./ V_k_norm;        
    end
    %
    for f = 1:length(freqs_sig)
        V_k_curr = V_k(:, :, f);        
        %%================= solve for vector c, (V * c = h)
        h_l = hobj_freq_grid.data(:, f, 1);
        h_r = hobj_freq_grid.data(:, f, 2);

        if sum(sum(isnan(V_k_curr)))
            c_BSM_l(:, f) = zeros(n_mic, 1);
            c_BSM_r(:, f) = zeros(n_mic, 1);
        else                
            % solution to Tikhonov regularization problem            
            if magLS && freqs_sig(f) >= f_cut_magLS                    
                % ===== MAG LS     
                if ~magLS_cvx
                    % Variable exchange method
                    [c_BSM_l(:, f), phase_last] = BSM_toolbox.TikhonovReg_MagLS_v2(V_k_curr, h_l, (1 / SNR_lin), phase_init_l_magLS, inv_opt, tol_magLS, max_iter_magLS);                                        
                    phase_init_l_magLS = phase_last;
                    [c_BSM_r(:, f), phase_last] = BSM_toolbox.TikhonovReg_MagLS_v2(V_k_curr, h_r, (1 / SNR_lin), phase_init_r_magLS, inv_opt, tol_magLS, max_iter_magLS);                                        
                    phase_init_r_magLS = phase_last;
                else
                    % SDP with CVX toolbox
                    c_BSM_l(:, f) = BSM_toolbox.TikhonovReg_MagLS_CVX(V_k_curr, h_l, (1 / SNR_lin), inv_opt);
                    c_BSM_r(:, f) = BSM_toolbox.TikhonovReg_MagLS_CVX(V_k_curr, h_r, (1 / SNR_lin), inv_opt);
                    % ==============
                end


            else
                % ===== CMPLX LS                    
                c_BSM_l(:, f) = BSM_toolbox.TikhonovReg(V_k_curr, h_l, (1 / SNR_lin), inv_opt);
                c_BSM_r(:, f) = BSM_toolbox.TikhonovReg(V_k_curr, h_r, (1 / SNR_lin), inv_opt);
                % ==============
            end                                
        end            
    end  
    
    
end




