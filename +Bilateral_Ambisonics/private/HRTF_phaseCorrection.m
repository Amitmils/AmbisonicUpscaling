% HRTF_phaseCorrection function to compute the phase corrected HRTF
%
% INPUTS:
%           hobj - HRTF in earo object in space domain
%           reconstructFlag - if true, then do reconstruction, i.e.
%                                       multiply by exp(ikr cos(Theta)
%                               else, multiply by exp(-ikr cos(Theta)
% 
% OUTPUT: 
%           hobj_pc - corrected hobj in space domain
%
% Written by Zamir Ben-Hur
% 18/12/18
function hobj_pc = HRTF_phaseCorrection(hobj, reconstructFlag, ra, th_ears, ph_ears)
    nfft = hobj.taps;
    c = 343; 
    fs = hobj.fs;
    f = (0:(nfft-1))*(fs/nfft); %linspace(0,fs/2,size(hobj.data,2));    % frequency [m/sec]
    f = f(1:end/2+1);
    k = 2*pi*f/c;             % wave number [rad/m]
    if strcmp(hobj.dataDomain{1}, 'TIME')
        hobj = hobj.toFreq(nfft);
        hobj.data = hobj.data(:,1:end/2+1,:);
    end

    if ~exist('ra','var') || isempty(ra)
        if ~isempty(hobj.micGrid.r)
            ra = hobj.micGrid.r; % radius of the head
        else
            ra = 0.0875;
        end
    end


    if ~exist('th_ears','var') || isempty(th_ears)
        th_ears=[pi/2,pi/2]; % locations of the ears
    end
    if ~exist('ph_ears','var') || isempty(ph_ears)
        ph_ears=[pi/2,3*pi/2];
    end
    
    theta = hobj.sourceGrid.elevation(:).';
    phi = hobj.sourceGrid.azimuth(:).';
%     phi(phi>pi) =  phi(phi>pi) - 2*pi;
    h_trans_l = zeros(size(hobj.data,1), size(hobj.data,2));
     h_trans_r = zeros(size(hobj.data,1), size(hobj.data,2));
    if length(th_ears(:,1))>1 % different ra,th_ears and ph_ears for each frequency
        for fInd = 1:size(hobj.data,2) % for every freq
            cosTheta_l = cos(theta)*cos(th_ears(fInd,1))+cos(phi-ph_ears(fInd,1)).*sin(theta)*sin(th_ears(fInd,1));
            cosTheta_r = cos(theta)*cos(th_ears(fInd,2))+cos(phi-ph_ears(fInd,2)).*sin(theta)*sin(th_ears(fInd,2));
            if reconstructFlag
                h_trans_l(:,fInd) = squeeze(hobj.data(:,fInd,1)) .* exp(1i*ra(fInd)*k(fInd)*cosTheta_l.');
                h_trans_r(:,fInd) = squeeze(hobj.data(:,fInd,2)) .* exp(1i*ra(fInd)*k(fInd)*cosTheta_r.');
            else
                h_trans_l(:,fInd) = squeeze(hobj.data(:,fInd,1)) .* exp(-1i*ra(fInd)*k(fInd)*cosTheta_l.');
                h_trans_r(:,fInd) = squeeze(hobj.data(:,fInd,2)) .* exp(-1i*ra(fInd)*k(fInd)*cosTheta_r.');
            end
        end
        
    elseif length(ra)>1 % only different ra for each frequency
        cosTheta_l = cos(theta)*cos(th_ears(1))+cos(phi-ph_ears(1)).*sin(theta)*sin(th_ears(1));
          cosTheta_r = cos(theta)*cos(th_ears(2))+cos(phi-ph_ears(2)).*sin(theta)*sin(th_ears(2));
         for fInd = 1:size(hobj.data,2) % for every freq
            if reconstructFlag
                h_trans_l(:,fInd) = squeeze(hobj.data(:,fInd,1)) .* exp(1i*ra(fInd)*k(fInd)*cosTheta_l.');
                h_trans_r(:,fInd) = squeeze(hobj.data(:,fInd,2)) .* exp(1i*ra(fInd)*k(fInd)*cosTheta_r.');
            else
                h_trans_l(:,fInd) = squeeze(hobj.data(:,fInd,1)) .* exp(-1i*ra(fInd)*k(fInd)*cosTheta_l.');
                h_trans_r(:,fInd) = squeeze(hobj.data(:,fInd,2)) .* exp(-1i*ra(fInd)*k(fInd)*cosTheta_r.');
            end
        end
    else
        cosTheta_l = cos(theta)*cos(th_ears(1))+cos(phi-ph_ears(1)).*sin(theta)*sin(th_ears(1));
        cosTheta_r = cos(theta)*cos(th_ears(2))+cos(phi-ph_ears(2)).*sin(theta)*sin(th_ears(2));
        hl = squeeze(hobj.data(:,:,1));
        hr = squeeze(hobj.data(:,:,2));
        for fInd = 1:size(hobj.data,2) % for every freq
            if reconstructFlag
                h_trans_l(:,fInd) = hl(:,fInd).* exp(1i*ra*k(fInd)*cosTheta_l.');
                h_trans_r(:,fInd) = hr(:,fInd) .* exp(1i*ra*k(fInd)*cosTheta_r.');
            else
                h_trans_l(:,fInd) = hl(:,fInd) .* exp(-1i*ra*k(fInd)*cosTheta_l.');
                h_trans_r(:,fInd) = hr(:,fInd) .* exp(-1i*ra*k(fInd)*cosTheta_r.');
            end
        end
    end
    hobj_pc = hobj.copy;
    hobj_pc.data(:,:,1) = h_trans_l;
    hobj_pc.data(:,:,2) = h_trans_r;
end