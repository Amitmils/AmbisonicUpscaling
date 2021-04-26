function PlotBSMfilters(BSMobj, c_BSM_time, domain)
arguments
    BSMobj struct
    c_BSM_time (:, :) double
    domain (1, 1) string {mustBeMember(domain, ["freq", "time"])} = 'freq'
end
%% PlotBSMfilters.m
% Plot BSM filters in either time or frequency domain
%% Inputs:
% BSMobj            : (MATLAB object) containing parameters
% c_BSM_time        : (n_mic x samples) BSM filters in time domain
% domain            : (string) either 'freq' or 'time'

    % init
    freqs_sig = BSMobj.freqs_sig;
    desired_fs = BSMobj.desired_fs;
    
    c_BSM_time_f = fft(c_BSM_time, [], 2);
    n_mic = size(c_BSM_time, 1);
    c_len = size(c_BSM_time, 2);
    
    % frequency domain plots
    if strcmp(domain, 'freq')
        for mm = 1:n_mic
            figure;
            subplot(2, 1, 1);            
            plot(freqs_sig, mag2db(abs(c_BSM_time_f(mm, 1:length(freqs_sig)))));
            title(['Filter for mic \#',num2str(mm)]);
            ylabel('Magnitude [dB]');            
            subplot(2, 1, 2);    
            plot(freqs_sig, angle(c_BSM_time_f(mm, 1:length(freqs_sig))));    
            ylabel('Phase [rad]');
            xlabel('Frequency [Hz]');            
        end
    elseif strcmp(domain, 'time')
        for mm = 1:n_mic
            figure;
            plot(linspace(0, 1, c_len) * c_len / desired_fs, (c_BSM_time(mm, :)));
            title(['Filter for mic \#',num2str(mm)]);
            ylabel('Amplitude');                     
            xlabel('Time [sec]');
        end
    end

end