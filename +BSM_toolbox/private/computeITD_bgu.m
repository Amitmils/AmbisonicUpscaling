%COMPUTEITD Compute ITD to move the signal A to match signal B
% INPUTS
%   A      : signal A (left HRIR)
%   B      : signal b (right HRIR)
%   fs     : sample rate (required for computing ITD in miliseconds and for method #4
%   method : method number (1 - Xcorr [Kulkarni 1999], 2- [Jot 1995], 3 - IACCe [Katz 2005], 4- Threshold detection [Andreopoulou and Katz 2017]
%
% OUTPUTS
%   tau_samples      : ITD in samples
%   tau_microseconds : ITD in microseconds
%
% NOTES
%   Last update: Zamir Ben-Hur 11.3.17
%
function [ tau_samples, tau_microseconds ] = computeITD_bgu( A, B, fs, method)
if ~exist('fs','var')
    fs = [];
end
if ~exist('method','var')  || isempty(method)
    warning('No ITD estimation method selected, using IACCe')
    method = 3;
end

tau_microseconds = [];
print = false;

switch method
    case 1 %% METHOD
        % [Kulkarni el al. 1999]
        [acor, lag] = xcorr(B, A);
        [~,index] = max(abs(acor));
        tau_samples = lag(index);
        
    case 2 %% METHOD 2 (UNSTABLE METHOD, DO NOT USE)
        % computing the ITD based on [Jot et al. Digital signal processsing issues in the context of binaural and transural sterephony]
        % BPF
        %{
        low_freq = 1000;
        high_freq = 5000;
        b = fir1(48,[low_freq/(fs/2) high_freq/(fs/2)]);
        delay = mean(grpdelay(b));
        
        A_bp = filter(b,1,A);
        A_bp(1:delay)=[];     %remove delay from filter
        B_bp = filter(b,1,B);
        B_bp(1:delay)=[];     %remove delay from filter
        %}
        
        % LPF
%         high_freq = 3000;
        high_freq = 1500;
        b = fir1(48,high_freq/(fs/2), 'low');
        delay = mean(grpdelay(b));
        
        A_lp = filter(b,1,A);
        A_lp(1:delay)=[];     %remove delay from filter
        B_lp = filter(b,1,B);
        B_lp(1:delay)=[];     %remove delay from filter
        
%         [acor, lag] = xcorr(B_bp, A_bp);
        [acor, lag] = xcorr(B_lp, A_lp);
        [~,index] = max(abs(acor));
%         [~,index] = max(acor);
        tau_samples = lag(index);
        
    case 3 %% METHOD 3:
        % IACCe [Katz et al. Subjective investigations of the interaural time difference in the horizontal plane]
        [A_up, ~] = envelope(A,100);
        [B_up, ~] = envelope(B,100);
        [acor, lag] = xcorr(B_up, A_up);
        [~,index] = max(abs(acor));
        tau_samples = lag(index);
        
    case 4 %% METHOD 4:
        % [Andreopoulou and Katz JASA 2017] Threshold detection method with -30dB, applied on 3kHz low-pass filtered HRIRs
        %
        % Method based on the detection of the time difference between the first onsets of the signals in the two ears.
        % Onsets are selected relative to predetermined level (-30 dB) thresholds as the first point in a signal
        % that exceeds that level value (Algazi et al., 2001).
        
        if isempty(fs)
           error('Error in computeITD: method #4 requires sample rate fs');
        end
        
        onset_threshold_dB = -30;
        
        % LPF
%         high_freq = 3000;
        high_freq = 1500;
        b = fir1(48,high_freq/(fs/2));
        delay = mean(grpdelay(b));
        
        A_lp = filter(b,1,A);
        A_lp(1:delay)=[];     %remove delay from filter
        B_lp = filter(b,1,B);
        B_lp(1:delay)=[];     %remove delay from filter
        
        % calculate linear onset threshold from dB value
%         onset_threshold = 10^(-onset_threshold_dB/20) ;  % error
        onset_threshold = 10^(onset_threshold_dB/20) ;
%         onset_threshold = 0.8;  % according to paper "Estimation of a Spherical-Head Model from Anthropometry"
        
        % find peaks and compute the sample position : A
        [maxA,iA ] = max(abs(A_lp)) ;
        kA = 0;
        
        while kA <= iA
            kA = kA +1;
%             if abs( A(kA)) > abs(maxA*onset_threshold )
            if abs( A_lp(kA)) > abs( maxA*onset_threshold )
                break ;
            end
        end
        if kA == 0
            error('Error in computeITD: signal A- Problem finding the onset \n') ;
            kA = 1;
        end
        
        % find peaks and compute the sample position : B
        [maxB,iB ] = max(abs(B_lp)) ;
        kB = 0;
        
        while kB <= iB
            kB = kB +1;
%             if abs( B(kB)) > abs(maxB*onset_threshold )
            if abs( B_lp(kB)) > abs( maxB*onset_threshold )
                break ;
            end      
        end
        if kB == 0
            error('Error in computeITD: signal B- : Problem finding the onset \n') ;
            kB = 1;
        end
        
        % calculate the ITD in seconds instead of samples
        tau_samples = (kB-kA);
        
    case 5 %% METHOD 5:
        
        % LPF
        high_freq = 1500;
        b = fir1(48,high_freq/(fs/2));
        delay = mean(grpdelay(b));
        
        A_lp = filter(b,1,A);
        A_lp(1:delay)=[];     %remove delay from filter
        B_lp = filter(b,1,B);
        B_lp(1:delay)=[];     %remove delay from filter
        
        
        t = (1:length(A_lp)).';
        A_center_of_energy = round(sum(t.*abs(A_lp).^2)/sum(abs(A_lp).^2));
        B_center_of_energy = round(sum(t.*abs(B_lp).^2)/sum(abs(B_lp).^2));
%         A_center_of_energy = round(sum(t.*A_lp)/sum(A_lp));
%         B_center_of_energy = round(sum(t.*B_lp)/sum(B_lp));
        
        tau_samples = (B_center_of_energy-A_center_of_energy);
        if (print == 1)
            doa = doa*(180/pi) -180;
            %h = figure(2);
            %subplot(2,1,1)
            plot(A_lp,'DisplayName','Left HRIR');
            hold on
            plot(B_lp,'DisplayName','Right HRIR');
            plot(A_center_of_energy,A_lp(A_center_of_energy),'r*')
            %hold off
            title("source position: "+string(doa));
            %subplot(2,1,2)
            %hold on
            plot(B_center_of_energy,B_lp(B_center_of_energy),'r*')
            hold off
            ylim([-0.3,0.3]);
            grid on
            legend
            %title("Right HRIR")
            drawnow()
            %gif
        end
end

if ~isempty(fs)
    tau_microseconds = tau_samples/fs * 1e6;
end

end

