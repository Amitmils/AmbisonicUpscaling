function [Pt_earAligned_sig,fs] =  Bilateral_Ambisonics_binaural_reproduction(anmt_l, anmt_r, fs, N, head_vec, HRTFpath,rot_ang,is_rot)
            

%set freqiemcy vectors
%------------------------------------------
c = 343;   %speed of sound


disp("anmt -> anmk")
tic
% Transform from time to frequency domain
%make length of x even, and calculate frequency range
nfft = max([size(anmt_l,1) size(anmt_r,1)]);
nfft = 2^nextpow2(nfft);
fftDim = 1;

anmk_l = fft(anmt_l,nfft,fftDim);
anmk_l = anmk_l.';          %SH X freq
anmk_l = anmk_l(:,1:end/2+1);

anmk_r = fft(anmt_r,nfft,fftDim);
anmk_r = anmk_r.';          %SH X freq
anmk_r = anmk_r(:,1:end/2+1);


f=linspace(0,fs/2,nfft/2+1);    % frequency range
w=2*pi*f;                       % radial frequency
k=w/c;                          %wave number
kr=k*head_vec(1);               %k*head_radius
toc

if is_rot
    disp('Rotate anms...')
    tic
    [anm_l_k_A,anm_r_k_A] = rotate_anms(anmk_l,anmk_r,rot_ang,N,head_vec,kr);
    toc
else
    anm_l_k_A = anmk_l;
    anm_r_k_A = anmk_r;
end

         

% Process HRTFs
disp('loading HRTFs and time -> freq...')
tic
% Load HRIR data
load(HRTFpath);
if hobj.fs~=fs
    hobj = hobj.resampleData(fs);
end
hobj.micGrid.r = head_vec(1);




if strcmp(hobj.dataDomain{1}, 'TIME')
    hobj = hobj.toFreq(nfft);             %this takes a long time
    hobj.data = hobj.data(:,1:end/2+1,:); %this takes a long time
    hobj.taps = nfft;
end
toc

% Calculating ear-aligned HRTF coefficients
disp('Calculating ear-aligned HRTF coefficients...');
tic
hobj_pc = HRTF_phaseCorrection(hobj, 0); %this takes a long time
toc

% Calculating SH matrix
disp('Ear-aligned HRTF Omega -> nm...')
tic
Y_ear = shmat(N,[hobj.sourceGrid.elevation.',hobj.sourceGrid.azimuth.'],true,true); % spherical harmonics matrix
pY_ear = pinv(Y_ear);

H_l_nm_pc = double(squeeze(hobj_pc.data(:,:,1)).')*pY_ear; % SH transform
H_r_nm_pc = double(squeeze(hobj_pc.data(:,:,2)).')*pY_ear; % SH transform
toc

disp('Calculating BRIRs...')
% Calculate Binaural singals in the SH domain - low order Bilateral
tild_N_ear_mat = tildize(N);
pad_ear = (N+1)^2;
anm_tilde_l = anm_l_k_A(:,1:size(H_l_nm_pc,1)).'*tild_N_ear_mat;
anm_tilde_r = anm_r_k_A(:,1:size(H_l_nm_pc,1)).'*tild_N_ear_mat;
pl_SH_earAligned = sum(padarray(anm_tilde_l,[0 pad_ear-size(anm_tilde_l,2)],'post').*padarray(H_l_nm_pc,[0 pad_ear-size(H_l_nm_pc,2)],'post'),2).';
pr_SH_earAligned = sum(padarray(anm_tilde_r,[0 pad_ear-size(anm_tilde_r,2)],'post').*padarray(H_r_nm_pc,[0 pad_ear-size(H_r_nm_pc,2)],'post'),2).';

% Calculate presure BRIR's
disp('Calculating P(t)...')
Pt_earAligned_sig   = calc_pt(pl_SH_earAligned.',pr_SH_earAligned.');

disp('Done')
end

%Functions
%--------------------------------
function [anm_l_k_A,anm_r_k_A] = rotate_anms(anmk_l,anmk_r,rot_ang,N,head_vec,kr)
    % make Wigner D
    % -------------------------------
    rot_ang = [0,0,rot_ang];
    D       = WignerDM(N,-rot_ang(1),-rot_ang(2),-rot_ang(3));
    % Rotate the head
    % -------------------------------
    [x0,y0,z0]  = s2c(head_vec(2),head_vec(3),head_vec(1)); %left ear vector in cartesians
    LeftEar     = [x0,y0,z0];
    inc = abs(LeftEar) > 1e-10;
    LeftEar = LeftEar.*inc;

    [x0,y0,z0]  = s2c(head_vec(4),head_vec(5),head_vec(1)); %right ear vector in cartesians
    RightEar    = [x0,y0,z0];
    inc = abs(RightEar) > 1e-10;
    RightEar = RightEar.*inc;

    head_vec_rotated = head_vec;
    head_vec_rotated(3) = head_vec_rotated(3) + rot_ang(3);
    head_vec_rotated(5) = head_vec_rotated(5) + rot_ang(3);

%     [x0,y0,z0]  = s2c(head_vec_rotated(2),head_vec_rotated(3),head_vec_rotated(1)); %left ear vector in cartesians
%     LeftEar_r     = [x0,y0,z0];
%     inc = abs(LeftEar_r) > 1e-10;
%     LeftEar_r = LeftEar_r.*inc;
% 
%     [x0,y0,z0]  = s2c(head_vec_rotated(4),head_vec_rotated(5),head_vec_rotated(1)); %right ear vector in cartesians
%     RightEar_r    = [x0,y0,z0];
%     inc = abs(RightEar_r) > 1e-10;
%     RightEar_r = RightEar_r.*inc;

    M = getRotationMat_ZYZ(-rot_ang(1),-rot_ang(2),-rot_ang(3));
    LeftEar_r_M = (M*LeftEar.').';
    RightEar_r_M = (M*RightEar.').';


    %Rotate anm's
    %---------------------------------------
    [a_grid,th_grid,ph_grid] = equiangle_sampling(N+3);
    %Y = sh2(N,th_grid,ph_grid);
    Y = shmat(N,[th_grid.',ph_grid.'],true,true); % spherical harmonics matrix

    Yp=diag(a_grid)*Y';

    anm_l_k_A       = anm_rotation(anmk_l,LeftEar,LeftEar_r_M,D,Y,Yp,th_grid,ph_grid,kr);
    anm_r_k_A       = anm_rotation(anmk_r,RightEar,RightEar_r_M,D,Y,Yp,th_grid,ph_grid,kr);

end

function anm_k_D_c = anm_rotation(anm_k,micPos,micPos_r,D,Y,Yp,th_grid,ph_grid,kr)

    anm_k_D = D*anm_k;              %SH/Freq (wigner D)
    a_k_D = Y.'*anm_k_D;            %Space/Freq

    e_mat     = build_e_mat_positive(micPos,micPos_r,th_grid,ph_grid,kr);

    a_k_D_c   = a_k_D.*e_mat;         %Space/Freq (phase correction)

    anm_k_D_c = (a_k_D_c.'*Yp).';   %SH/Freq



end
function e_mat = build_e_mat_positive(micPos,micPos_r,th_grid,ph_grid,kr)
    th_s = th_grid;
    ph_s = ph_grid;
    
    [th_e,ph_e,~] = c2s(micPos(1),micPos(2),micPos(3));
    [th_e_r,ph_e_r,~] = c2s(micPos_r(1),micPos_r(2),micPos_r(3));

    COS_OMEGA_0 = cos(th_s)*cos(th_e)+cos(ph_s - ph_e).*sin(th_s)*sin(th_e); %phase correction
    COS_OMEGA_1 = cos(th_s)*cos(th_e_r)+cos(ph_s - ph_e_r).*sin(th_s)*sin(th_e_r); %phase correction

    d1          = (COS_OMEGA_0 - COS_OMEGA_1);
    e_mat       = exp(1i*d1.'*kr);

end

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
function Pt = calc_pt(Pl,Pr)
    Pl(1)=real(Pl(1));                   Pr(1)=real(Pr(1));
    Pl(end)=real(Pl(end));               Pr(end)=real(Pr(end));
    Pl=[Pl;flipud(conj(Pl(2:end-1)))];   Pr=[Pr;flipud(conj(Pr(2:end-1)))];
    
    Pl=real(ifft(Pl,'symmetric')); %zero pedding length of data
    Pr=real(ifft(Pr,'symmetric'));
    Pt = [Pl,Pr];
end
