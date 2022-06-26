function MagLS_CVX_test(BSMobj, V_k, hobj_freq_grid)

%init
tol_magLS       = BSMobj.tol_magLS;
max_iter_magLS  = BSMobj.max_iter_magLS;    
normSV          = BSMobj.normSV;
SNR_lin         = BSMobj.SNR_lin;
inv_opt         = BSMobj.inv_opt;    

%normalize steering vectors    
if normSV
    V_k_norm = vecnorm(V_k, 2, 1);
    V_k = V_k ./ V_k_norm;        
end

f = 100;
% SNR_lin = 10^8;
inv_opt = 2;

V_k_curr = V_k(:, :, f);        
%%================= solve for vector c, (V * c = h)
h_l = hobj_freq_grid.data(:, f, 1);

% ===== MAG LS     
phase_init_l_magLS = pi / 2;    % according to [1]
[c_BSM_magLS_l, ~] = BSM_toolbox.TikhonovReg_MagLS_v2(V_k_curr, h_l, (1 / SNR_lin), phase_init_l_magLS, inv_opt, tol_magLS, max_iter_magLS);                                        
err_magLS = sqrt(norm(abs(V_k_curr' * c_BSM_magLS_l) - abs(h_l), 2)^2 / (norm(conj(h_l), 2)^2));
% ==============

% ===== MAG LS with CVX     
c_BSM_magLScvx_l = BSM_toolbox.TikhonovReg_MagLS_CVX(V_k_curr, h_l, (1 / SNR_lin), inv_opt);
err_magLScvx = sqrt(norm(abs(V_k_curr' * c_BSM_magLScvx_l) - abs(h_l), 2)^2 / (norm(conj(h_l), 2)^2));
% ==============

% ===== CMPLX LS                    
c_BSM_cmplx_l = BSM_toolbox.TikhonovReg(V_k_curr, h_l, (1 / SNR_lin), inv_opt);
err_cmlpx = sqrt(norm(abs(V_k_curr' * c_BSM_cmplx_l) - abs(h_l), 2)^2 / (norm(conj(h_l), 2)^2));
% ==============

fprintf('Mag LS error: %.2f\n', err_magLS);
fprintf('Mag LS CVX error: %.2f\n', err_magLScvx);
fprintf('Complex LS error: %.2f\n', err_cmlpx);

end