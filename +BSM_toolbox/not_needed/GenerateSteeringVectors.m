function V_k = GenerateSteeringVectors(BSMobj, th_DOA, ph_DOA)
%% GenerateSteeringVectors.m
% Generate steering vectors that are used in BSM filters 
%% Inputs:
% BSMobj            : (MATLAB object) containing parameters
% th_DOA            : (1 x DOAs) elevation of arriving/incident plane waves [rad]
% ph_DOA            : (1 x DOAs) azimuth of arriving/incident plane waves [rad]
%% Outputs:
% V_k               : (freq x DOAs x M) steering vectors in freq. domain

    %init
    r_array         = BSMobj.r_array;
    N_PW            = BSMobj.N_PW;
    freqs_sig       = BSMobj.freqs_sig;    
    c               = BSMobj.c;    
    th_array        = BSMobj.th_array;
    ph_array        = BSMobj.ph_array;    
    normSV          = BSMobj.normSV;    
    rigidArray      = BSMobj.rigidArray;
    M               = BSMobj.M;    
            
    % calculation of SV with SH: exp = Ymic * B * Y'
    N_SV = N_PW;
    Ymic   = sh2(N_SV, th_array, ph_array).';  % with copensation [mic x SH]  
    Y = conj(sh2(N_SV, th_DOA, ph_DOA));  % [SH x PWs]
        
    V_k = zeros(length(freqs_sig), M, length(th_DOA));
    for f = 1:length(freqs_sig)
        lambda = c / freqs_sig(f); % wave-length
        kr = 2 * pi / lambda * r_array;
        %N_SV = ceil(kr)+1;
        %N_SV = 30;        
        Bn_mat = BnMat(N_SV, kr, kr, rigidArray);                    
        V_k(f, :, :)   = (Ymic * diag(Bn_mat) * Y);                     

        %normalize steering vectors
        if normSV
            V_k_norm = sqrt(sum(abs(V_k(f, :, :)).^2, 2));
            V_k(f, :, :) = V_k(f, :, :) ./ V_k_norm;
        end
                                  
    end  
    
    V_k = permute(V_k, [1 3 2]);    
    
    
end




