function [Pt_earAligned_sig,fs] =  Bilateral_Ambisonics_binaural_reproduction(anmt_l, anmt_r, fs, N, head_vec, HRTFpath,rot_ang)
            

%set freqiemcy vectors
%------------------------------------------
%fs = 48e3;
%nfft= 2^11;
c = 343;   %speed of sound


 


% Load HRIR data
load(HRTFpath);
if hobj.fs~=fs
    hobj = hobj.resampleData(fs);
end
hobj.micGrid.r = head_vec(1);

disp('Rotate anms...')
% Transform from time to frequency domain
%make length of x even, and calculate frequency range
anmt_r      = anmt_r.';
anmt_l      = anmt_l.';
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

[anm_l_k_A,anm_r_k_A,anmk_c_ref] = rotate_anms(anmk_l,anmk_r,...
    rot_ang,N,head_vec,kr);

         


% Calculating SH matrix
disp('Calculating SH matrices...')

%Y_ear = sh2(N,hobj.sourceGrid.elevation,hobj.sourceGrid.azimuth); % spherical harmonics matrix
Y_ear = shmat(N,[hobj.sourceGrid.elevation.',hobj.sourceGrid.azimuth.'],true,true); % spherical harmonics matrix

pY_ear = pinv(Y_ear);

%Y_inf = sh2(N_ref,hobj.sourceGrid.elevation,hobj.sourceGrid.azimuth); % spherical harmonics matrix infinity order
Y_inf = shmat(N_ref,[hobj.sourceGrid.elevation.',hobj.sourceGrid.azimuth.'],true,true); % spherical harmonics matrix infinity order

pY_inf = pinv(Y_inf);

% Process HRTFs
disp('Calculating SHT of HRTFs...')
if strcmp(hobj.dataDomain{1}, 'TIME')
    hobj = hobj.toFreq(nfft);
    hobj.data = hobj.data(:,1:end/2+1,:);
    hobj.taps = nfft;
end

% Calculating HRTF coefficients - high order
H_l_nm_full = double(squeeze(hobj.data(:,:,1)).')*pY_inf; % SH transform
H_r_nm_full = double(squeeze(hobj.data(:,:,2)).')*pY_inf; % SH transform

% Calculating ear-aligned HRTF coefficients
disp('Calculating ear-aligned HRTF coefficients...');
hobj_pc = HRTF_phaseCorrection(hobj, 0);
H_l_nm_pc = double(squeeze(hobj_pc.data(:,:,1)).')*pY_ear; % SH transform
H_r_nm_pc = double(squeeze(hobj_pc.data(:,:,2)).')*pY_ear; % SH transform
    
% Calculate Binaural singals in the SH domain - high order
disp('Calculating BRIRs...')
tild_N_inf_mat = tildize(N_ref);
pad_center = (N_ref+1)^2;

anm_tilde_full = anmk_c_ref.'*tild_N_inf_mat;

pl_SH_full = sum(padarray(anm_tilde_full,[0 pad_center-size(anm_tilde_full,2)],'post').*padarray(H_l_nm_full,[0 pad_center-size(H_l_nm_full,2)],'post'),2).';
pr_SH_full = sum(padarray(anm_tilde_full,[0 pad_center-size(anm_tilde_full,2)],'post').*padarray(H_r_nm_full,[0 pad_center-size(H_r_nm_full,2)],'post'),2).';

% Calculate Binaural singals in the SH domain - low order Bilateral
tild_N_ear_mat = tildize(N);
pad_ear = (N+1)^2;

% anm_tilde_l = anmk_l(:,1:size(H_l_nm_pc,1)).'*tild_N_ear_mat;
% anm_tilde_r = anmk_r(:,1:size(H_l_nm_pc,1)).'*tild_N_ear_mat;
anm_tilde_l = anm_l_k_A(:,1:size(H_l_nm_pc,1)).'*tild_N_ear_mat;
anm_tilde_r = anm_r_k_A(:,1:size(H_l_nm_pc,1)).'*tild_N_ear_mat;
pl_SH_earAligned = sum(padarray(anm_tilde_l,[0 pad_ear-size(anm_tilde_l,2)],'post').*padarray(H_l_nm_pc,[0 pad_ear-size(H_l_nm_pc,2)],'post'),2).';
pr_SH_earAligned = sum(padarray(anm_tilde_r,[0 pad_ear-size(anm_tilde_r,2)],'post').*padarray(H_r_nm_pc,[0 pad_ear-size(H_r_nm_pc,2)],'post'),2).';

% Calculate presure BRIR's
disp('Calculating BRIR P(t)...')
Pt_full         = calc_pt(pl_SH_full.',pr_SH_full.');
Pt_earAligned   = calc_pt(pl_SH_earAligned.',pr_SH_earAligned.');

% Convolve dry signal with the the room impulse responcess
disp('Calculating P(t)...')
Pt_full_left = BRIR2pt(Pt_full(:,1),s);
Pt_full_right = BRIR2pt(Pt_full(:,2),s);
Pt_full_sig =[Pt_full_left,Pt_full_right];

Pt_earAligned_left = BRIR2pt(Pt_earAligned(:,1),s);
Pt_earAligned_right = BRIR2pt(Pt_earAligned(:,2),s);
Pt_earAligned_sig =[Pt_earAligned_left,Pt_earAligned_right];

disp('Done')
end

%Functions
%--------------------------------
function [anm_l_k_A,anm_r_k_A] = rotate_anms(anmk_l,anmk_r,rot_ang,N,head_vec,kr)
    % make Wigner D
    % -------------------------------
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

    [x0,y0,z0]  = s2c(head_vec_rotated(2),head_vec_rotated(3),head_vec_rotated(1)); %left ear vector in cartesians
    LeftEar_r     = [x0,y0,z0];
    inc = abs(LeftEar_r) > 1e-10;
    LeftEar_r = LeftEar_r.*inc;

    [x0,y0,z0]  = s2c(head_vec_rotated(4),head_vec_rotated(5),head_vec_rotated(1)); %right ear vector in cartesians
    RightEar_r    = [x0,y0,z0];
    inc = abs(RightEar_r) > 1e-10;
    RightEar_r = RightEar_r.*inc;

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
