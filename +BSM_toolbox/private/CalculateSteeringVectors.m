function V_k = CalculateSteeringVectors(BSMobj, N_SV, th_DOA, ph_DOA)
%% CalculateSteeringVectors.m
% Calculate steering vectors of supported arrays (spherical / semi-circular / fully-circular)
% Equation for calculation is: Ymic * B * Y'
%% Inputs:
% BSMobj            : (MATLAB object) containing parameters
% N_SV              : (scalar) maximal SH order of calculated steering vectors
% th_DOA            : (DOAs x 1) elevation of arriving/incident plane waves [rad]
% ph_DOA            : (DOAs x 1) azimuth of arriving/incident plane waves   [rad]
%% Outputs:
% V_k               : (n_mic x Q x freq) steering vectors (ATF)
% V_k               : (freq x DOAs x M) steering vectors in freq. domain

    %init
    r_array         = BSMobj.r_array;    
    freqs_sig       = BSMobj.freqs_sig;    
    c               = BSMobj.c;
    n_mic           = BSMobj.n_mic;                
    th_array        = BSMobj.th_array;
    ph_array        = BSMobj.ph_array;        
    normSV          = BSMobj.normSV;
    sphereType      = BSMobj.sphereType;    
    
    if size(th_DOA, 2) > size(th_DOA, 1)
        th_DOA = th_DOA .';
    end    
    if size(ph_DOA, 2) > size(ph_DOA, 1)
        ph_DOA = ph_DOA .';
    end

    f_len = length(freqs_sig);
    N_len = (N_SV + 1)^2;
    DOAs_len = length(th_DOA);
    Y = conj(shmat(N_SV, [th_DOA, ph_DOA], true, true));            % [SH x PWs]
    Ymic   = shmat(N_SV, [th_array.', ph_array.'], true, false);    % [mic x SH]  
    
    kr = 2 * pi * r_array * freqs_sig / c;
    b = bn(N_SV, kr, "sphereType", sphereType);            
    bb = repmat(b, 1, 1, DOAs_len);
    YY = repmat(Y, 1, 1, f_len); YY = permute(YY, [3 1 2]);
    bY = bb .* YY;
    bY1 = permute(bY, [2 1 3]); 
    bY2 = reshape(bY1, N_len, f_len * DOAs_len);
    V_k = Ymic * bY2;
    V_k = reshape(V_k, n_mic, f_len, DOAs_len);
    V_k = permute(V_k, [2 3 1]);
    % steering vector is [n_mic x directions x f_len]
    
    %normalize steering vectors    
    if normSV
        V_k_norm = vecnorm(V_k, 2, 3);
        V_k = V_k ./ V_k_norm;        
    end
    
end