function p_array_f = GenerateArrayMeasurements(BSMobj, anm_f)
%%GenerateArrayMeasurements.m
% Generate array measurements - only supported arrays in BSM study!
%%Inputs:
% BSMobj            : (MATLAB object) containing parameters
% anm_f             : ((N_PW + 1)^2 x DOAs) SFT coefficients of sound field (PWD)
%%Outputs:
% p_array_f         : (freq x DOAs x mics) array recordings (sound pressure at mics) in frequency domain

    % init
    freqs_sig =     BSMobj.freqs_sig;
    ph_DOA =        BSMobj.ph_DOA;
    n_mic =         BSMobj.n_mic;
    N_PW =          BSMobj.N_PW;
    th_array =      BSMobj.th_array;
    ph_array =      BSMobj.ph_array;
    c =             BSMobj.c;
    r_array =       BSMobj.r_array;
    rigidArray =    BSMobj.rigidArray;    
    
    %% ================= calculate array measurements
    % calculate steering vector with SH: exp = Ymic * B * Y'
    p_array_f = zeros(length(freqs_sig), length(ph_DOA), n_mic);  % p_array_f is (freq) x (DOA of PW) x (n_mic)    
    %     N_SV = min([N_PW, ceil(kr)+1]);
    %     N_SV = 30;
    N_SV = N_PW;
    Ymic = sh2(N_SV, th_array, ph_array).';  % [mic x SH]
    for f = 1:length(freqs_sig)
        lambda = c / freqs_sig(f); % wave-length
        kr = 2 * pi / lambda * r_array;
        Bn_mat = BnMat(N_SV, kr, kr, rigidArray);
        
        p_array_f(f, :, :) = permute(Ymic * diag(Bn_mat) * anm_f(1:(N_SV + 1)^2, :), [2 1]);
    end
end