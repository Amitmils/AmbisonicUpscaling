close all
clear
%addpath(genpath('/Users/orberebi/Dropbox/Matlab/bilateral_room_sim/'));
%addpath(genpath('/Users/orberebi/Documents/AClab/ACLtoolbox'));
%addpath(genpath('/Users/orberebi/Documents/GitHub/PHALCOR/toolbox/'));

%
%Simulation Parameters
%----------------------
N_ref = 40;                  %high order standart reproduction
N = 4;                        %Synthesis SH order
c=343;                      %speed of sound
refCoef=0.75;               %reflecation coeff
roomDims=[8,6,4];           % room dimensions
r_0 = 0.0875;               %8.75cm head radius
recPos=[4,3,1.7];           % Mic position (head center)
%srcPos = [6.2,4.7,1.7];     %source position
srcPos = [6,3,1.7];     %source position
rot_ang = [0,0,-30*(pi/180)]; %[alpha,beta,gamma]

%Plot room and recording geometry
%=================================================
plot_room(recPos,srcPos,roomDims)
drawnow()


%Load dry signal
%----------------------------------
filename = "/Users/orberebi/Documents/GitHub/general/+Bilateral_Ambisonics/Dry_signals/casta.wav";
[s,fs] = audioread(filename);   %fs is the sample rate of the .wav file
fs_sim = 196e3;                 %the image methoud sample rate

filename = "/Users/orberebi/Documents/GitHub/general/+Bilateral_Ambisonics/HRTF/earoHRIR_KU100_Measured_2702Lebedev.mat";
hrtf = load(filename);


[Pt_full_sig,Pt_earAligned_sig] =  bilateral_room_sim_func(N_ref, N,...
    c,refCoef, roomDims, r_0, recPos, srcPos,s,fs,fs_sim,hrtf.hobj,rot_ang);




%Main functions
%--------------------------------
function [Pt_full_sig,Pt_earAligned_sig] =  bilateral_room_sim_func(N_ref, N, c,refCoef, roomDims, r_0, recPos, srcPos,s,fs,fs_sim,hobj,rot_ang)

%set freqiemcy vectors
%------------------------------------------
%fs = 48e3;
%nfft= 2^11;



%head size position and orientation
%----------------------------------
th_0_l = (pi/180)*(90);                             %left ear position
ph_0_l = (pi/180)*(90);                             %left ear position
th_0_r = (pi/180)*(90);                             %right ear position
ph_0_r = (pi/180)*(270);                            %right ear position
head_vec = [r_0,th_0_l,ph_0_l,th_0_r,ph_0_r];

[x0,y0,z0]=s2c(head_vec(2),head_vec(3),head_vec(1));
recPosL = [x0,y0,z0] + recPos;
[x0,y0,z0]=s2c(head_vec(4),head_vec(5),head_vec(1));
recPosR = [x0,y0,z0] + recPos;
 


% Load HRIR data

if hobj.fs~=fs
    hobj = hobj.resampleData(fs);
end
hobj.micGrid.r = r_0;

gComDiv = gcd(fs_sim, fs);
p = double(fs / gComDiv);
q = double(fs_sim / gComDiv);

%capture anm head center pre rotation N reference
%------------------------------------------------
disp('Capture anm head center...')
[anmt_c_ref,~, roomParams] = get_anm(fs_sim, N_ref, roomDims, refCoef, srcPos, recPos, p,q);     %SH X freq and SH X time

%capture anm Left ear
%---------------------------------------
disp('Capture anm Left ear...')
[anmt_l,~, ~] = get_anm(fs_sim, N, roomDims, refCoef, srcPos, recPosL, p,q);     %SH X freq and SH X time
%capture anm Right ear
%---------------------------------------
disp('Capture anm Right ear...')
[anmt_r,~, ~] = get_anm(fs_sim, N, roomDims, refCoef, srcPos, recPosR, p,q);     %SH X freq and SH X time
   

% Transform from time to frequency domain
%make length of x even, and calculate frequency range
anmt_c_ref  = anmt_c_ref.';
anmt_r      = anmt_r.';
anmt_l      = anmt_l.';
nfft = max([size(anmt_l,1) size(anmt_r,1) size(anmt_c_ref,2)]);
if mod(nfft,2)
    nfft = nfft+1;
end
fftDim = 1;

anmk_l = fft(anmt_l,nfft,fftDim);
anmk_l = anmk_l.';          %SH X freq
anmk_l = anmk_l(:,1:end/2+1);

anmk_r = fft(anmt_r,nfft,fftDim);
anmk_r = anmk_r.';          %SH X freq
anmk_r = anmk_r(:,1:end/2+1);

anmk_c_ref = fft(anmt_c_ref,nfft,fftDim);
anmk_c_ref = anmk_c_ref.';  %SH X freq
anmk_c_ref = anmk_c_ref(:,1:end/2+1);

f=linspace(0,fs/2,nfft/2+1);    % frequency range
w=2*pi*f;                       % radial frequency
c=343;                          % speed of sound
k=w/c;                          %wave number
kr=k*head_vec(1);               %k*head_radius


% make Wigner D
% -------------------------------
D       = WignerDM(N,-rot_ang(1),-rot_ang(2),-rot_ang(3));
D_ref   = WignerDM(N_ref,-rot_ang(1),-rot_ang(2),-rot_ang(3));
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
Y=sh2(N,th_grid,ph_grid);
Yp=diag(a_grid)*Y';

anm_l_k_A       = anm_rotation(anmk_l,LeftEar,LeftEar_r_M,D,Y,Yp,th_grid,ph_grid,kr);
anm_r_k_A       = anm_rotation(anmk_r,RightEar,RightEar_r_M,D,Y,Yp,th_grid,ph_grid,kr);
anmk_c_ref      = D_ref*anmk_c_ref;    %left ear post rotation true mesurments

         


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


% % Calculating ear-aligned HRTF coefficients
% disp('Calculating ear-aligned HRTF coefficients...')
% hobj_pc = calc_ear_hrtf_v6(hobj,head_vec,N,fs,c);
% % hobj_pc = HRTF_phaseCorrection(hobj, 0);
% H_l_nm_pc = double(squeeze(hobj_pc.data(:,:,1)).')*pY_ear; % SH transform
% H_r_nm_pc = double(squeeze(hobj_pc.data(:,:,2)).')*pY_ear; % SH transform


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
function Pt = calc_pt(Pl,Pr)
    Pl(1)=real(Pl(1));                   Pr(1)=real(Pr(1));
    Pl(end)=real(Pl(end));               Pr(end)=real(Pr(end));
    Pl=[Pl;flipud(conj(Pl(2:end-1)))];   Pr=[Pr;flipud(conj(Pr(2:end-1)))];
    
    Pl=real(ifft(Pl,'symmetric')); %zero pedding length of data
    Pr=real(ifft(Pr,'symmetric'));
    Pt = [Pl,Pr];
end
function ab = BRIR2pt(a,b)
ab = fftfilt([a;zeros(length(b)-1,1)],[b;zeros(length(a)-1,1)]);

% NFFT = size(BRIR,1)+size(sig,1)-1;
% 
% 
% 
% Pl = fft(BRIR(:,1),NFFT);            Pr = fft(BRIR(:,2),NFFT);
% Pl(1)=real(Pl(1));                   Pr(1)=real(Pr(1));
% Pl(end)=real(Pl(end));               Pr(end)=real(Pr(end));
% Pl=[Pl;flipud(conj(Pl(2:end-1)))];   Pr=[Pr;flipud(conj(Pr(2:end-1)))];
% 
% Pl=real(ifft(Pl,'symmetric')); %zero pedding length of data
% Pr=real(ifft(Pr,'symmetric'));
% Pt = [Pl,Pr];
%     
%     
% % Internal use only
% 
% NFFT = size(a,1)+size(b,1)-1;
% A    = fft(a,NFFT);
% B    = fft(b,NFFT);
% AB   = A.*B;
% ab   = ifft(AB);
end
function [anmt_out,parametric, roomParams] = get_anm(fs_sim, N, roomDims,refCoef, srcPos, recPos, p, q)
%input:     simulation sample rate, order, room parameters, source position, 
%           mic position, p and q for down-sampeling, frequncy resolution
%==========================================
    %simulate with tom's image methoud
    %======================================
    [anmt, parametric, roomParams] = image_method.calc_rir(fs_sim, roomDims,...
    srcPos, recPos, refCoef, {"angle_dependence", false,"max_reflection_order", 300}, ...
    {"array_type", "anm", "N", N,"bpfFlag",true});
    anmt_out = anmt.';
    %Down-sample anmt
    %======================================
    anm_N_tmp = zeros(ceil(size(anmt,1)* (p/q)), size(anmt,2));
    for rInd = 1:size(anmt,2)
        anm_N_tmp(:,rInd) = resample( anmt(:,rInd),p,q);
    end
    anmt = anm_N_tmp;
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

function plot_room(recPos,srcPos_des,roomDims)
%Plot room and recording geometry
%------------------------------------------------
[th0,ph0,r0]=c2s(srcPos_des(1)-recPos(1),srcPos_des(2)-recPos(2),srcPos_des(3)-recPos(3));
ph0=mod(ph0,2*pi);
srcPos_relative=[r0,th0,ph0];
roomSimulationPlot_ISF(roomDims, srcPos_relative, recPos) %from ACLtoolbox
end