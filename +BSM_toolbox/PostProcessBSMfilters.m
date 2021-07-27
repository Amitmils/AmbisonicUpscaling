function [c_BSM_l_time_cs, c_BSM_r_time_cs] = PostProcessBSMfilters(BSMobj, c_BSM_l, c_BSM_r)
%%PostProcessBSMfilters.m
% This function post-process BSM filters given in the frequency domain:
% 1. compute conjugate symmetric 
% 2. iFFT
% 3. circular shift (add delay) to generate the non-causal part of the filter at the beginning
%%Inputs:
% BSMobj    : (MATLAB object) containing parameters
% c_BSM_l   : (n_mic x freq) BSM filters for left ear in freq. domain
% c_BSM_r   : (n_mic x freq) BSM filters for right ear in freq. domain
% sig_samp  : (scalar) number of samples of signal/HRTF
%%Outputs:
% c_BSM_l_time_cs : (n_mic x sig_samp) Post-processed BSM filters for left ear in time domain
% c_BSM_l_time_cs : (n_mic x sig_samp) Post-processed BSM filters for right ear in time domain
    
    filt_samp = BSMobj.filt_samp;
%     filt_samp = 0;
        
    % 1. compute conjugate symmetric (not really necessary, can use zeros)
    %{
    if ~mod(filt_samp, 2)
       % filter length is even    
       c_BSM_l(:, 1) = real(c_BSM_l(:, 1));
       c_BSM_l(:, end) = real(c_BSM_l(:, end));
       c_BSM_l_sym = [c_BSM_l, conj(c_BSM_l(:, end-1:-1:2))];

       c_BSM_r(:, 1) = real(c_BSM_r(:, 1));
       c_BSM_r(:, end) = real(c_BSM_r(:, end));
       c_BSM_r_sym = [c_BSM_r, conj(c_BSM_r(:, end-1:-1:2))];
    else
       % filter length is odd       
       c_BSM_l(:, 1) = real(c_BSM_l(:, 1));
       c_BSM_l_sym = [c_BSM_l, conj(c_BSM_l(:, end:-1:2))];

       c_BSM_r(:, 1) = real(c_BSM_r(:, 1));
       c_BSM_r_sym = [c_BSM_r, conj(c_BSM_r(:, end:-1:2))];       
    end
    %}
    
    % 1. Add zeros to correct filter length
    %
    c_BSM_l_sym = c_BSM_l;
    c_BSM_r_sym = c_BSM_r;
    c_BSM_l_sym(:, end+1 : filt_samp) = 0;
    c_BSM_r_sym(:, end+1 : filt_samp) = 0;
    %}    
    
    % 2. iFFT       
    c_BSM_l_time = ifft(c_BSM_l_sym, [], 2, 'symmetric');
    c_BSM_r_time = ifft(c_BSM_r_sym, [], 2, 'symmetric');    
    %t_vec = (0:length(c_BSM_l_time) - 1) ./ desired_fs;
    
    % Conjugate filter in freq domain -> time revervsal
    %{
    c_BSM_l_time = [c_BSM_l_time(:, 1), c_BSM_l_time(:, end:-1:2)];
    c_BSM_r_time = [c_BSM_r_time(:, 1), c_BSM_r_time(:, end:-1:2)];
    %}
       
    % 3. circular shift (add delay) to generate the non-causal part of the filter at the beginning    
    %
    if ~mod(filt_samp, 2)
        % filter length is even  
        c_BSM_l_time_cs = circshift(c_BSM_l_time, filt_samp / 2, 2);
        c_BSM_r_time_cs = circshift(c_BSM_r_time, filt_samp / 2, 2);        
    else
        % filter length is odd
        c_BSM_l_time_cs = circshift(c_BSM_l_time, (filt_samp - 1) / 2, 2);
        c_BSM_r_time_cs = circshift(c_BSM_r_time, (filt_samp - 1) / 2, 2);        
    end
    %}    
    
end





