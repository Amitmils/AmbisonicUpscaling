%% This script studies HRTF ITD/ILD estimation errors with BSM (complex and magnitude LS versions)

% Date created: November 24, 2020
% Created by:   Lior Madmoni
% Modified :    April 12, 2021    

clearvars;
close all;
clc;

restoredefaultpath;
% add ACLtoolbox path
addpath(genpath('/Users/orberebi/Documents/GitHub/general'));
%cd('/Users/liormadmoni/Google Drive/Lior/Acoustics lab/Matlab/Research/Github/general/');

startup_script();
rng('default');

% add AKtoolbox to path (from old ACLtoolbox in shared GoogleDrive)
addpath(genpath('/Users/orberebi/Documents/GitHub/Binaural_Cues_Optimization/functions/AKtools/'));
% add export_fig to path
%addpath(genpath('/Volumes/GoogleDrive/My Drive/Lior/Acoustics lab/Matlab/Research/FB_BFBR/Toolboxes/altmany-export_fig-9aba302'));

% parameters/flags - array
filt_len = 0.032;                                      % filters (BSM/HRTF) length [sec]
arrayType = 1;                                         % 0 - spherical array, 1 - semi-circular array, 2 - full-circular array
rigidArray = 1;                                        % 0 - open array, 1 - rigid array
M = 6;                                                 % number of microphones
r_array = 0.1;                                         % array radius
head_rot_az = ...
    wrapTo2Pi(deg2rad([0, 30, 60]));                     % vector of head rotations [degrees]
normSV = true;                                         % true - normalize steering vectors

% parameters/flags - general
c = 343;                                               % speed of sound [m/s]
desired_fs = 48000;                                    % choose samplong frequency in Hz
N_PW = 30;                                             % SH order of plane-wave synthesis
saveFiles = false;                                     % save MATLAB files before time interpolation?
save_plot_flag = false;                                % save plots locally

% parameters/flags - BSM design
inv_opt = 1;                                           % opt=1 -> ( (1 / lambda) * (A * A') + eye )  |||| opt2=1 -> ((A * A') + lambda * eye);
source_distribution = 1;                               % 0 - nearly uniform (t-design), 1 - spiral nearly uniform
Q = 240;                                               % Assumed number of sources
f_cut_magLS = 1500;                                    % above cutoff frequency use MagLS
tol_magLS = 1e-20;    
max_iter_magLS = 1E5;
%noise
SNR = 20;                                              % assumed sensors SNR [dB]
sig_n = 0.1;
sig_s = 10^(SNR/10) * sig_n;
SNR_lin = sig_s / sig_n;    
%signal
filt_samp    = filt_len * desired_fs;
freqs_sig    = ( 0 : (filt_samp / 2) ) * desired_fs / filt_samp;
freqs_sig(1) = 1/4 * freqs_sig(2); %to not divide by zero

% Text variables for plots 
if ~rigidArray
    sphereType = 'open';
else
    sphereType = 'rigid';
end
switch arrayType 
    case 0
        arrayTypeTxt = [sphereType,'Spherical'];
    case 1
        arrayTypeTxt = [sphereType,'SemiCirc'];
    case 2
        arrayTypeTxt = [sphereType,'FullCirc'];
end

% ILD/ITD params
COMPUTE_ITD = true;     % flag whether or not to compute ITD
COMPUTE_ILD = true;    % flag whether or not to compute ILD
% Incident plane-wave (to study ITD/ILD)
ph_DOA_res = 1;         % Azimuth resolution in degrees
ph_DOA_max = 360;       % maximal azimuth in degrees
ph_DOA = deg2rad(0:ph_DOA_res:ph_DOA_max);
th_DOA = deg2rad(90 * ones(size(ph_DOA)));
%
methodITD = 2;
switch methodITD
    case 1
        ITDmethod_title = 'method-1';
    case 2
        ITDmethod_title = 'corr method';
    case 3
        ITDmethod_title = 'IACCe method';
    case 4
        ITDmethod_title = 'Threshold method';
    case 5
        ITDmethod_title = 'CenterOfMass method';
end
high_freq_ITD = 1500;
saveITDvid = false;
% fig_for_ITD = figure('visible','off', 'Position', [1, 1, 1280, 800]);
% ILD_flim = [1500 20e3]; ILD_freq_bands = 22;
% ILD_flim = [50 20e3]; ILD_freq_bands = 39;
ILD_flim = [50 6000]; ILD_freq_bands = 21;
% ILD_flim = [50 6000]; ILD_freq_bands = 27;
ILD_fmax_ave = 6000;
if saveITDvid
    mkdir(['plots/ITD_ILD_err/ITD_vid/']);
    fig_for_ITD = figure('visible','off', 'Position', [1, 1, 1280, 800]);
    % choose video length in seconds
    vidLen = 10;
    % open video write object     
    vidName = ['BSM_',arrayTypeTxt,'_M=',num2str(M),'_highFreq=',num2str(high_freq_ITD),'_SNR=',num2str(SNR)];
    ITDvid_file = VideoWriter(['plots/ITD_ILD_err/ITD_vid/',vidName], 'MPEG-4');
    ITDvid_file.FrameRate = max([1, round(length(ph_DOA) / vidLen)]);
    open(ITDvid_file);
else
    ITDvid_file = [];
    fig_for_ITD = [];
end
%
if COMPUTE_ITD
    ITD_ref = zeros(length(ph_DOA), length(head_rot_az), length(M));
    ITD_BSM_cmplx = zeros(length(ph_DOA), length(head_rot_az), length(M));
    ITD_BSM_mag = zeros(length(ph_DOA), length(head_rot_az), length(M));
end
if COMPUTE_ILD
    ILD_ref = zeros(length(ph_DOA), length(head_rot_az), length(M), ILD_freq_bands);
    ILD_BSM_cmplx = zeros(length(ph_DOA), length(head_rot_az), length(M), ILD_freq_bands);
    ILD_BSM_mag = zeros(length(ph_DOA), length(head_rot_az), length(M), ILD_freq_bands);
end

%% ================= HRTFS preprocessing
% load HRIRs
N_HRTF = 30;
HRTFpath =  '/Users/orberebi/Documents/GitHub/Binaural_Cues_Optimization/HRTF/KU100/earoHRIR_KU100_Measured_2702Lebedev.mat';
% HRTFpath =  '/Users/liormadmoni/Google Drive/ACLtoolbox/Data/HRTF/earoHRIR_KEMAR_TU_BEM_OnlyHead.mat';
load(HRTFpath);         % hobj is HRIR earo object
hobj.shutUp = false;
[th_BSMgrid_vec, ph_BSMgrid_vec] = BSM_toolbox.BSMgrid(source_distribution, Q);
%%Interpolate HRTF to frequencies
hobj_freq_grid = hobj;
if strcmp(hobj_freq_grid.dataDomain{1},'FREQ'), hobj_freq_grid=hobj_freq_grid.toTime(); end
% resample HRTF to desired_fs
hobj_freq_grid = hobj_freq_grid.resampleData(desired_fs);
hobj_freq_grid = hobj_freq_grid.toFreq(filt_samp);
% Trim negative frequencies
hobj_freq_grid.data = hobj_freq_grid.data(:, 1:ceil(filt_samp/2)+1, :);

%% ================= Load WignerD Matrix
WignerDpath = '/Users/orberebi/Documents/GitHub/general/+examples/data/WignerDMatrix_diagN=32.mat';
load(WignerDpath);
N_HRTF_rot = 30;
DN = (N_HRTF_rot + 1)^2; % size of the wignerD matrix
D_allAngles = D(:, 1 : DN);

%% ================= Create BSM struct
BSMobj.freqs_sig = freqs_sig;
BSMobj.ph_DOA = ph_DOA;    
BSMobj.N_PW = N_PW;    
BSMobj.c = c;
BSMobj.r_array = r_array;
BSMobj.rigidArray = rigidArray;
BSMobj.th_BSMgrid_vec = th_BSMgrid_vec;
BSMobj.ph_BSMgrid_vec = ph_BSMgrid_vec;
%
BSMobj.f_cut_magLS = f_cut_magLS;
BSMobj.tol_magLS = tol_magLS;
BSMobj.max_iter_magLS = max_iter_magLS;
BSMobj.normSV = normSV;
BSMobj.SNR_lin = SNR_lin;
BSMobj.inv_opt = inv_opt;
BSMobj.head_rot_az = head_rot_az;
BSMobj.M = M;
BSMobj.Q = Q;
BSMobj.source_distribution = source_distribution;
BSMobj.desired_fs = desired_fs;
BSMobj.filt_samp = filt_samp;
BSMobj.sphereType = sphereType;

for m = 1:length(M)
    %% ================= Get array positions
    n_mic = M(m);        
    [th_array, ph_array, ~] = BSM_toolbox.GetArrayPositions(arrayType, n_mic, 0);       
    
    %% ================= Update BSM struct
    BSMobj.n_mic = n_mic;
    BSMobj.th_array = th_array;
    BSMobj.ph_array = ph_array;      
    
    %% ================= calculate array measurements
    BSMobj.normSV = false;
    p_array_f = CalculateSteeringVectors(BSMobj, N_PW, th_DOA.', ph_DOA.');
    % [p_array_f] = (freq x DOAs x mics) 
    
    % Transform p_array_f to time domain    
    % pad negative frequencies with zeros (has no effect since we use ifft with "symmetric" flag)
    p_array_f(end+1 : filt_samp, :, :) = 0;
    p_array_t = ifft(p_array_f, [], 1, 'symmetric');
    p_array_t = circshift(p_array_t, round((size(p_array_t, 1) - 1) / 2), 1);        
    % [p_array_t] = (time x DOAs x mics) array recordings (sound pressure at mics) in time domain    
    clear p_array_f;
    fprintf('Finished calculating array measurements\n');
    
    %% ================= calculate array steering vectors (for BSM filters)
    N_SV = N_PW;
    BSMobj.normSV = false;
    V_k = CalculateSteeringVectors(BSMobj, N_SV, th_BSMgrid_vec, ph_BSMgrid_vec); 
    V_k = permute(V_k, [3 2 1]);    
    
    for h=1:length(head_rot_az)
        %% ================= Rotate HRTFs according to head rotation - new        
        hobj_rot = RotateHRTF(hobj_freq_grid, N_HRTF_rot, D_allAngles, head_rot_az(h));
        % Interpolate HRTF to BSM grid
        hobj_rot_BSM = hobj_rot;
        hobj_rot_BSM = hobj_rot_BSM.toSpace('SRC', th_BSMgrid_vec, ph_BSMgrid_vec);          

        %% ================= BSM method
        %%======Generate BSM filters in frequency domain
        %BSMobj.ph_array = ph_rot_array;
        BSMobj.normSV = normSV;
        % Complex version
        BSMobj.magLS = false;
        BSMobj.magLS_cvx = false;
        [c_BSM_cmplx_l, c_BSM_cmplx_r] = BSM_toolbox.GenerateBSMfilters_faster(BSMobj, V_k, hobj_rot_BSM);
        
        % MagLS version
        BSMobj.magLS = true;
        [c_BSM_mag_l, c_BSM_mag_r] = BSM_toolbox.GenerateBSMfilters_faster(BSMobj, V_k, hobj_rot_BSM);
        
        %%======Post-processing BSM filters (time domain)
        [c_BSM_cmplx_l_time_cs, c_BSM_cmplx_r_time_cs] = ...
            BSM_toolbox.PostProcessBSMfilters(BSMobj, c_BSM_cmplx_l, c_BSM_cmplx_r);
        [c_BSM_mag_l_time_cs, c_BSM_mag_r_time_cs] = ...
            BSM_toolbox.PostProcessBSMfilters(BSMobj, c_BSM_mag_l, c_BSM_mag_r);
        
        %%======Optional - plot filters in freq domain   
        %BSM_toolbox.PlotBSMfilters(BSMobj, c_BSM_cmplx_l_time_cs, 'time');
        
        %%======Filter microphone signals
        p_BSM_cmplx_t_l = zeros(filt_samp + size(p_array_t, 1) - 1, length(ph_DOA));
        p_BSM_cmplx_t_r = zeros(filt_samp + size(p_array_t, 1) - 1, length(ph_DOA));
        p_BSM_mag_t_l = zeros(filt_samp + size(p_array_t, 1) - 1, length(ph_DOA));
        p_BSM_mag_t_r = zeros(filt_samp + size(p_array_t, 1) - 1, length(ph_DOA));                
        
        % Direct filtering in frequency domain
        %{
        nfft = filt_samp + size(p_array_t, 1) - 1;
        c_BSM_cmplx_l_time_cs_f = fft(c_BSM_cmplx_l_time_cs, nfft, 2);
        c_BSM_cmplx_r_time_cs_f = fft(c_BSM_cmplx_r_time_cs, nfft, 2);
        c_BSM_mag_l_time_cs_f = fft(c_BSM_mag_l_time_cs, nfft, 2);
        c_BSM_mag_r_time_cs_f = fft(c_BSM_mag_r_time_cs, nfft, 2);
        for p = 1:length(ph_DOA)
            p_array_t_cur = squeeze(p_array_t(:, p, :));  % [DOA of PW x receiver]
            p_array_f_cur = fft(p_array_t_cur, nfft, 1);
            
            p_tmp = squeeze(sum(conj(c_BSM_cmplx_l_time_cs_f.') .* p_array_f_cur, 2));
            p_BSM_cmplx_t_l(:, p) = ifft(p_tmp, nfft, 1, 'symmetric');
            
            p_tmp = squeeze(sum(conj(c_BSM_cmplx_r_time_cs_f.') .* p_array_f_cur, 2));
            p_BSM_cmplx_t_r(:, p) = ifft(p_tmp, nfft, 1, 'symmetric');
            
            p_tmp = squeeze(sum(conj(c_BSM_mag_l_time_cs_f.') .* p_array_f_cur, 2));
            p_BSM_mag_t_l(:, p) = ifft(p_tmp, nfft, 1, 'symmetric');
            
            p_tmp = squeeze(sum(conj(c_BSM_mag_r_time_cs_f.') .* p_array_f_cur, 2));
            p_BSM_mag_t_r(:, p) = ifft(p_tmp, nfft, 1, 'symmetric');
        end
        %}
        
        %%Direct filtering in time domain - using fftfilt
        %
        % Time reversal - to conjugate filters in frequency domain        
        c_BSM_cmplx_l_time_cs = [c_BSM_cmplx_l_time_cs(:, 1), c_BSM_cmplx_l_time_cs(:, end:-1:2)];
        c_BSM_cmplx_r_time_cs = [c_BSM_cmplx_r_time_cs(:, 1), c_BSM_cmplx_r_time_cs(:, end:-1:2)];
        c_BSM_mag_l_time_cs = [c_BSM_mag_l_time_cs(:, 1), c_BSM_mag_l_time_cs(:, end:-1:2)];
        c_BSM_mag_r_time_cs = [c_BSM_mag_r_time_cs(:, 1), c_BSM_mag_r_time_cs(:, end:-1:2)];        
        for p = 1:length(ph_DOA)            
            p_array_t_cur = squeeze(p_array_t(:, p, :));  % [time/freq x DOAs x receiver]
            % zero-padding to correct length!
            p_array_t_cur = [p_array_t_cur; zeros(filt_samp - 1, n_mic)];
            %
            p_BSM_cmplx_t_l(:, p) = (sum(fftfilt(c_BSM_cmplx_l_time_cs.', p_array_t_cur), 2));
            p_BSM_cmplx_t_r(:, p) = (sum(fftfilt(c_BSM_cmplx_r_time_cs.', p_array_t_cur), 2));
            p_BSM_mag_t_l(:, p) = (sum(fftfilt(c_BSM_mag_l_time_cs.', p_array_t_cur), 2));
            p_BSM_mag_t_r(:, p) = (sum(fftfilt(c_BSM_mag_r_time_cs.', p_array_t_cur), 2));
        end                           
        %}
        
        % Direct filtering in time domain - manual implementation of fftfilt
        %{
        c_BSM_cmplx_l_time_cs_f = fft(c_BSM_cmplx_l_time_cs, [], 2);
        c_BSM_cmplx_r_time_cs_f = fft(c_BSM_cmplx_r_time_cs, [], 2);
        c_BSM_mag_l_time_cs_f = fft(c_BSM_mag_l_time_cs, [], 2);
        c_BSM_mag_r_time_cs_f = fft(c_BSM_mag_r_time_cs, [], 2);
        for p = 1:length(ph_DOA)
            p_array_t_cur = squeeze(p_array_t(:, p, :));  % [DOA of PW x receiver]
            p_array_f_cur = fft(p_array_t_cur, [], 1);
            
            p_tmp = conj(c_BSM_cmplx_l_time_cs_f.') .* p_array_f_cur;
            p_tmp = ifft(p_tmp, [], 1, 'symmetric');
            p_BSM_cmplx_t_l(:, p) = sum(p_tmp, 2);
            
            p_tmp = conj(c_BSM_cmplx_r_time_cs_f.') .* p_array_f_cur;
            p_tmp = ifft(p_tmp, [], 1, 'symmetric');
            p_BSM_cmplx_t_r(:, p) = sum(p_tmp, 2);
            
            p_tmp = conj(c_BSM_mag_l_time_cs_f.') .* p_array_f_cur;
            p_tmp = ifft(p_tmp, [], 1, 'symmetric');
            p_BSM_mag_t_l(:, p) = sum(p_tmp, 2);
            
            p_tmp = conj(c_BSM_mag_r_time_cs_f.') .* p_array_f_cur;
            p_tmp = ifft(p_tmp, [], 1, 'symmetric');
            p_BSM_mag_t_r(:, p) = sum(p_tmp, 2);
        end                        
        %}


        
        %% Analyze ITD/ILD  
        %==== Reference HRTF from hobj
        %
        HRTFs_hobj_f_all = hobj_rot;
        % interpolate HRTF to new grid 
        HRTFs_hobj_f_all = HRTFs_hobj_f_all.toSpace('SRC', th_DOA, ph_DOA);
        HRTFs_hobj_f_all = squeeze(HRTFs_hobj_f_all.data);   
        HRTFs_hobj_f_all = permute(HRTFs_hobj_f_all, [2, 1, 3]);
        % pad negative frequencies with zeros (has no effect since we use ifft with "symmetric" flag)
        HRTFs_hobj_f_all(end+1 : filt_samp, :, :) = 0;
        REF_sig = ifft(HRTFs_hobj_f_all, [], 1, 'symmetric');
        REF_sig = circshift(REF_sig, filt_samp / 2, 1);
        %}
        
        %==== BSM sig - cmplx
        BSM_cmplx_sig = cat(3, p_BSM_cmplx_t_l, p_BSM_cmplx_t_r);        
       
        %==== BSM sig - magnitude
        BSM_mag_sig = cat(3, p_BSM_mag_t_l, p_BSM_mag_t_r);        
                
        for p = 1:length(ph_DOA)            
            % first, normalize signals
%             bin_sig_ref_curr = squeeze(REF_sig(:, p, :, h));
            bin_sig_ref_curr = squeeze(REF_sig(:, p, :));
            bin_sig_ref_curr = 0.9 * bin_sig_ref_curr / max(max(abs(bin_sig_ref_curr)));
            bin_sig_BSM_cmplx_curr = squeeze(BSM_cmplx_sig(:, p, :));
            bin_sig_BSM_cmplx_curr = 0.9 * bin_sig_BSM_cmplx_curr / max(max(abs(bin_sig_BSM_cmplx_curr)));
            bin_sig_BSM_mag_curr = squeeze(BSM_mag_sig(:, p, :));
            bin_sig_BSM_mag_curr = 0.9 * bin_sig_BSM_mag_curr / max(max(abs(bin_sig_BSM_mag_curr)));
            
            %==== ITD calculation            
            if COMPUTE_ITD
                % ITD without video
                %{
                [~, ITD_ref(p, h, m) ] = computeITD_bgu( bin_sig_ref_curr(:, 1),...
                    bin_sig_ref_curr(:, 2), desired_fs, methodITD);
                [~, ITD_BSM_cmplx(p, h, m) ] = computeITD_bgu( bin_sig_BSM_cmplx_curr(:, 1),...
                    bin_sig_BSM_cmplx_curr(:, 2), desired_fs, methodITD);
                [~, ITD_BSM_mag(p, h, m) ] = computeITD_bgu( bin_sig_BSM_mag_curr(:, 1),...
                    bin_sig_BSM_mag_curr(:, 2), desired_fs, methodITD);
                %}
                
                % ITD with video
                %
                [~, ITD_ref(p, h, m) ] = computeITD_bgu_OrVersion( bin_sig_ref_curr(:, 1),...
                    bin_sig_ref_curr(:, 2), desired_fs, methodITD, high_freq_ITD, 0, ph_DOA(p), fig_for_ITD, saveITDvid, ITDvid_file);
                [~, ITD_BSM_cmplx(p, h, m) ] = computeITD_bgu_OrVersion( bin_sig_BSM_cmplx_curr(:, 1),...
                    bin_sig_BSM_cmplx_curr(:, 2), desired_fs, methodITD, high_freq_ITD, 0, ph_DOA(p), fig_for_ITD, saveITDvid, ITDvid_file);
                [~, ITD_BSM_mag(p, h, m) ] = computeITD_bgu_OrVersion( bin_sig_BSM_mag_curr(:, 1),...
                    bin_sig_BSM_mag_curr(:, 2), desired_fs, methodITD, high_freq_ITD, 0, ph_DOA(p), fig_for_ITD, saveITDvid, ITDvid_file);
                %}
            end
                
            %==== ILD calculation
            response_len    = length(freqs_sig);
            if COMPUTE_ILD
                Pl_ref_ILD = (bin_sig_ref_curr(:, 1));
                Pr_ref_ILD = (bin_sig_ref_curr(:, 2));
                %{
                if ~mod(response_len, 2)
                    % filter length is even  
                    Pl_ref_ILD = circshift(Pl_ref_ILD, response_len / 2, 2);
                    Pr_ref_ILD = circshift(Pr_ref_ILD, response_len / 2, 2);
                else
                    % filter length is odd
                    Pl_ref_ILD = circshift(Pl_ref_ILD, (response_len - 1) / 2, 2);
                    Pr_ref_ILD = circshift(Pr_ref_ILD, (response_len - 1) / 2, 2);
                end
                %}
                %Pl_ref_ILD   = circshift(Pl_ref_ILD, [round(size(Pl_ref_ILD,1)/2) 0]);
                %Pr_ref_ILD    = circshift(Pr_ref_ILD, [round(size(Pr_ref_ILD,1)/2) 0]);
                Pl_BSM_cmplx_ILD = bin_sig_BSM_cmplx_curr(:, 1);
                Pr_BSM_cmplx_ILD = bin_sig_BSM_cmplx_curr(:, 2);
                %Pl_BSM_cmplx_ILD   = circshift(Pl_BSM_cmplx_ILD, [round(size(Pl_BSM_cmplx_ILD,1)/2) 0]);
                %Pr_BSM_cmplx_ILD    = circshift(Pr_BSM_cmplx_ILD, [round(size(Pr_BSM_cmplx_ILD,1)/2) 0]);
                Pl_BSM_mag_ILD = bin_sig_BSM_mag_curr(:, 1);
                Pr_BSM_mag_ILD = bin_sig_BSM_mag_curr(:, 2);
                %Pl_BSM_mag_ILD   = circshift(Pl_BSM_mag_ILD, [round(size(Pl_BSM_mag_ILD,1)/2) 0]);
                %Pr_BSM_mag_ILD    = circshift(Pr_BSM_mag_ILD, [round(size(Pr_BSM_mag_ILD,1)/2) 0]);
                
                [ILD_ref_ERB, f_c_ERB, ~] = AKerbILD(Pl_ref_ILD, Pr_ref_ILD, ILD_flim, desired_fs);
                [ILD_BSM_cmplx_ERB, ~, ~] = AKerbILD(Pl_BSM_cmplx_ILD, Pr_BSM_cmplx_ILD, ILD_flim, desired_fs);
                [ILD_BSM_mag_ERB, ~, ~] = AKerbILD(Pl_BSM_mag_ILD, Pr_BSM_mag_ILD, ILD_flim, desired_fs);      

                ILD_ref(p, h, m, :) = ILD_ref_ERB;
                ILD_BSM_cmplx(p, h, m, :) = ILD_BSM_cmplx_ERB;
                ILD_BSM_mag(p, h, m, :) = ILD_BSM_mag_ERB;
            end
        end
        
        %% Analyze error in HRTF estimation - not needed here
        %{
        %==== Reference HRTF from hobj
        HRTFs_hobj_f_all = hobj;
        % interpolate HRTF to new grid
        HRTFs_hobj_f_all = HRTFs_hobj_f_all.toSH(N_HRTF, 'SRC');
        HRTFs_hobj_f_all = HRTFs_hobj_f_all.toSpace('SRC', th_DOA, ph_DOA);
        HRTFs_hobj_f_all = squeeze(HRTFs_hobj_f_all.data);
        for p=1:length(ph_DOA)
            %==== Reference HRTF from binaural signal of a single PW
            HRTFs_ref_t = squeeze(bin_sig_rot_t.data(h, :, p, :));
            %             HRTFs_ref_t = 0.9 * HRTFs_ref_t / max(max(abs(HRTFs_ref_t)));
            HRTFs_ref_f = fft(HRTFs_ref_t, [], 1);
            HRTFs_ref_f = HRTFs_ref_f(1:ceil(nFFT/2)+1, :);
            %             HRTFs_ref_f = HRTFs_ref_f / max(max(abs(HRTFs_ref_f)));
            
            %==== Reference HRTF from hobj
            HRTFs_hobj_f = squeeze(HRTFs_hobj_f_all(p, :, :, :));
            
            %==== Estimated HRTF by BSM
            HRTFs_BSM_f = [p_TRest_l(:, p), p_TRest_r(:, p)];
            %             HRTFs_BSM_f = HRTFs_BSM_f / max(max(abs(HRTFs_BSM_f)));
            
            % Optional HRTF plots
            % ======= left ear
            figure('Position', [1, 1, 1280, 800]);
            % magnitude - HRTF
            subplot(2, 1, 1);
            plot(freqs_sig, mag2db(abs(HRTFs_ref_f(:, 1))), 'linewidth', 2);
            hold on; grid on;
            plot(freqs_sig, mag2db(abs(HRTFs_hobj_f(:, 1))), 'linewidth', 2);
            plot(freqs_sig, mag2db(abs(HRTFs_BSM_f(:, 1))), 'linewidth', 2);
            xlim([75, 2e4]);
            ylabel('$|$HRTF$|$', 'interpreter', 'latex');
            title({[LS_title, ' $N_{\mathrm{PW}}=',num2str(N_PW),', N_{\mathrm{HRTF}}=',num2str(N_HRTF),'$'], ['$h_l(f, \theta=',num2str(rad2deg(th_DOA(p))),'^{\circ}, \phi=',...
                num2str(rad2deg(ph_DOA(p))),'^{\circ})$']}, 'interpreter', 'latex');
            legend('ref-bin', 'ref-hobj', 'BSM', 'location', 'best');
            if plt_log_scale
                set(gca, 'Xscale', 'log');
            end
            set(gca, 'fontsize', 25, 'linewidth', 2);
            % phase - HRTF
            subplot(2, 1, 2);
            plot(freqs_sig, angle(HRTFs_ref_f(:, 1)), 'linewidth', 2);
            hold on; grid on;
            plot(freqs_sig, angle(HRTFs_hobj_f(:, 1)), 'linewidth', 2);
            plot(freqs_sig, angle(HRTFs_BSM_f(:, 1)), 'linewidth', 2);
            xlim([75, 2e4]);
            ylabel('$\angle$ HRTF', 'interpreter', 'latex');
            xlabel('Freq [Hz]');
            legend('ref-bin', 'ref-hobj', 'BSM', 'location', 'best');
            if plt_log_scale
                set(gca, 'Xscale', 'log');
            end
            set(gca, 'fontsize', 25, 'linewidth', 2);
            %savefig
            if save_plot_flag
                mkdir(['plots/ITD_ILD_err/HRTFerr/',arrTypeFold,'/M=',num2str(n_mic),'/']);
                export_fig(['plots/ITD_ILD_err/HRTFerr/',arrTypeFold,'/M=',num2str(n_mic),'/hl_Npw=',num2str(N_PW),...
                    '_N_HRTF=',num2str(N_HRTF),'_phDOA=',num2str(rad2deg(ph_DOA(p)), '%d'),...
                    '_headRot=',num2str(rad2deg(head_rot_az(h)), '%d'),'_',LS_title(1:5),'.png'], '-transparent', '-r300');
            end
            
            % ======= right ear
            figure('Position', [1, 1, 1280, 800]);
            % magnitude - HRTF
            subplot(2, 1, 1);
            plot(freqs_sig, mag2db(abs(HRTFs_ref_f(:, 2))), 'linewidth', 2);
            hold on; grid on;
            plot(freqs_sig, mag2db(abs(HRTFs_hobj_f(:, 2))), 'linewidth', 2);
            plot(freqs_sig, mag2db(abs(HRTFs_BSM_f(:, 2))), 'linewidth', 2);
            xlim([75, 2e4]);
            ylabel('$|$HRTF$|$', 'interpreter', 'latex');
            title({[LS_title, ' $N_{\mathrm{PW}}=',num2str(N_PW),', N_{\mathrm{HRTF}}=',num2str(N_HRTF),'$'], ['$h_r(f, \theta=',num2str(rad2deg(th_DOA(p))),'^{\circ}, \phi=',...
                num2str(rad2deg(ph_DOA(p))),'^{\circ})$']}, 'interpreter', 'latex');
            legend('ref-bin', 'ref-hobj', 'BSM', 'location', 'best');
            if plt_log_scale
                set(gca, 'Xscale', 'log');
            end
            set(gca, 'fontsize', 25, 'linewidth', 2);
            % phase - HRTF
            subplot(2, 1, 2);
            plot(freqs_sig, angle(HRTFs_ref_f(:, 2)), 'linewidth', 2);
            hold on; grid on;
            plot(freqs_sig, angle(HRTFs_hobj_f(:, 2)), 'linewidth', 2);
            plot(freqs_sig, angle(HRTFs_BSM_f(:, 2)), 'linewidth', 2);
            xlim([75, 2e4]);
            ylabel('$\angle$ HRTF', 'interpreter', 'latex');
            xlabel('Freq [Hz]');
            legend('ref-bin', 'ref-hobj', 'BSM', 'location', 'best');
            if plt_log_scale
                set(gca, 'Xscale', 'log');
            end
            set(gca, 'fontsize', 25, 'linewidth', 2);
            %savefig
            if save_plot_flag
                mkdir(['plots/ITD_ILD_err/HRTFerr/',arrTypeFold,'/M=',num2str(n_mic),'/']);
                export_fig(['plots/ITD_ILD_err/HRTFerr/',arrTypeFold,'/M=',num2str(n_mic),'/hr_Npw=',num2str(N_PW),...
                    '_N_HRTF=',num2str(N_HRTF),'_phDOA=',num2str(rad2deg(ph_DOA(p)), '%d'),...
                    '_headRot=',num2str(rad2deg(head_rot_az(h)), '%d'),'_',LS_title(1:5),'.png'], '-transparent', '-r300');
            end
            
        end
        
        %}
        
        fprintf('Finished ITD/ILD calculation for mic idx = %d/%d, head rotation idx = %d/%d\n'...
            ,m, length(M), h, length(head_rot_az));
        
    end
    
end

% close video object
if saveITDvid
    close(ITDvid_file);
end

% set maximal middle frequency of ERB filter for ILD averaging
if COMPUTE_ILD
    f_c_max_ERB_ind = f_c_ERB < ILD_fmax_ave;
    ILD_fc_title = ['Maximal ERB freq=',num2str(ILD_fmax_ave/1000, '%.1f'),' kHz'];
    %
    ILD_ref_av       = mean(ILD_ref(:, :, :, f_c_max_ERB_ind), 4);
    ILD_BSM_cmplx_av = mean(ILD_BSM_cmplx(:, :, :, f_c_max_ERB_ind), 4);
    ILD_BSM_mag_av   = mean(ILD_BSM_mag(:, :, :, f_c_max_ERB_ind), 4);
    %
    ILD_BSM_cmplx_av_err = mean(abs(ILD_ref(:, :, :, f_c_max_ERB_ind) - ILD_BSM_cmplx(:, :, :, f_c_max_ERB_ind)), 4);     
    ILD_BSM_mag_av_err   = mean(abs(ILD_ref(:, :, :, f_c_max_ERB_ind) - ILD_BSM_mag(:, :, :, f_c_max_ERB_ind)), 4); 
    %
    ILD_BSM_cmplx_freq_err = abs(ILD_ref - ILD_BSM_cmplx); 
    ILD_BSM_mag_freq_err   = abs(ILD_ref - ILD_BSM_mag); 
    
    % ERB filter bandwidth calc
    ERB_BW = 24.7 * (0.00437 * f_c_ERB + 1);
    f_c_max_effective_freq = f_c_ERB + 0.5 * ERB_BW;
end

if COMPUTE_ITD
    ITD_cmplx_err    = abs(ITD_ref - ITD_BSM_cmplx);
    ITD_mag_err      = abs(ITD_ref - ITD_BSM_mag);
end

%% Plots - ITD
c_order = num2cell(get(gca,'colororder'), 2);
if COMPUTE_ITD
    % ITD error as func of M
    for h = 1:length(head_rot_az)
        figure('Position', [1, 1, 800, 400]);
        % ITD subplot
        subplot(2, 1, 1);    
        plot(rad2deg(ph_DOA), squeeze(ITD_ref(:, h, 1)), 'linewidth', 3, 'color', c_order{1}); hold on; grid on;    
        plot(rad2deg(ph_DOA), squeeze(ITD_BSM_cmplx(:, h, :)), 'linewidth', 3, 'color', c_order{2});
        plot(rad2deg(ph_DOA), squeeze(ITD_BSM_mag(:, h, :)), 'linewidth', 3, 'color', c_order{3});
        xlim([0 ph_DOA_max]);
        ylabel('ITD [$\mu s$]', 'interpreter', 'latex');
        %title({[arr_txt, ' with head rotation = $',num2str(rad2deg(head_rot_az(h))),'^{\circ}$'],...
        %    [ITDmethod_title]},'interpreter', 'latex');
        %M_leg_cmplx = cellstr(num2str(M.'));
        %M_leg_cmplx = strcat('$M=',M_leg_cmplx,'$ cmplx');
        %M_leg_mag = cellstr(num2str(M.'));
        %M_leg_mag = strcat('$M=',M_leg_mag,'$ mag');
        %M_leg_all = [{'Ref'}; M_leg_cmplx; M_leg_mag];
        %legend(M_leg_all, 'interpreter', 'latex');
        leg_all = [{'Reference'}; {'BSM'}; {'BSM-MagLS'}];
        legend(leg_all, 'location', 'best', 'interpreter', 'latex');
        set(gca, 'fontsize', 20, 'fontname', 'times', 'linewidth', 2);  
        set(gca,'xticklabel',[]);

        % ITD error subplot
        subplot(2, 1, 2);
        plot(rad2deg(ph_DOA), squeeze(ITD_cmplx_err(:, h, :)), 'linewidth', 3, 'color', c_order{2}); hold on; grid on;    
        plot(rad2deg(ph_DOA), squeeze(ITD_mag_err(:, h, :)), 'linewidth', 3, 'color', c_order{3});
        xlim([0 ph_DOA_max]);
        ylim([0 600]);
        %xlabel('Azimuth [deg]', 'interpreter', 'latex');
        ylabel('$\epsilon_{\mathrm{ITD}}$ [$\mu s$]', 'interpreter', 'latex');    
        %M_leg_cmplx = cellstr(num2str(M.'));
        %M_leg_cmplx = strcat('$M=',M_leg_cmplx,'$ cmplx');
        %M_leg_mag = cellstr(num2str(M.'));
        %M_leg_mag = strcat('$M=',M_leg_mag,'$ mag');
        %M_leg_all = [M_leg_cmplx; M_leg_mag];
        %legend(M_leg_all, 'interpreter', 'latex');
        leg_all = [{'BSM'}; {'BSM-MagLS'}];        
        legend(leg_all, 'location', 'best', 'interpreter', 'latex');
        set(gca, 'fontsize', 20, 'fontname', 'times', 'linewidth', 2);   
        
        % position
        ha=get(gcf,'children');
        %set(ha(1),'position',[.1 .5 1 0.5])
        set(ha(2),'position',[.1 .1 .87 .4])
        %set(ha(3),'position',[.1 .5 1 0.5])
        set(ha(4),'position',[.1 .55 .87 .4])

        %savefig
        if save_plot_flag
            %mkdir(['plots/ITD_ILD_err/',arrTypeFold,'/M/']);
            %export_fig(['plots/ITD_ILD_err/',arrTypeFold,'/M/ITD_err_Npw=',num2str(N_PW),'_',rigidArrTxt,'_Q=',num2str(Q_spiral),'_ra=',num2str(r_array),'_headRot=',num2str(rad2deg(head_rot_az(h)), '%d'),'_cmplxVSmag.png'], '-transparent', '-r300');
            
            export_fig(['/Users/liormadmoni/Google Drive/Lior/Acoustics lab/'...
                ,'Research/FB/Binaural_beamforming/Journal_paper/figs/ITD/'...
                ,'ITD_err_Npw=',num2str(N_PW),'_',sphereType,'_headRot=',num2str(rad2deg(head_rot_az(h)), '%d'),'_cmplxVSmag.png'], '-transparent', '-r300');
        end
    end

    % ITD error as func of head_rot
    %{
    for m = 1:length(M)    
        figure('Position', [1, 1, 1280, 800]);        
        % ITD subplot
        subplot(2, 1, 1);
        plot(rad2deg(ph_DOA), squeeze(ITD_ref(:, :, 1)), 'linewidth', 2); hold on; grid on;    
        plot(rad2deg(ph_DOA), squeeze(ITD_BSM(:, :, m)), 'linewidth', 2);
        xlim([0 ph_DOA_max]);
        ylabel('ITD [$\mu s$]', 'interpreter', 'latex');
        title({[arr_txt,' with M = $',num2str(M(m)),'$'],...
            [ITDmethod_title]},'interpreter', 'latex');
        head_rot_leg = cellstr(num2str(rad2deg(head_rot_az).'));
        head_rot_leg1 = strcat('Ref head rot =$',head_rot_leg,'^{\circ}$');
        head_rot_leg2 = strcat('BSM Head rot =$',head_rot_leg,'^{\circ}$');
        head_rot_leg = [head_rot_leg1; head_rot_leg2];
        legend(head_rot_leg, 'interpreter', 'latex');
        set(gca, 'fontsize', 25, 'linewidth', 2);    

        % ITD error subplot
        subplot(2, 1, 2);
        plot(rad2deg(ph_DOA), squeeze(ITD_err(:, :, m)), 'linewidth', 2); hold on; grid on;    
        xlim([0 ph_DOA_max]);
        xlabel('Azimuth [deg]', 'interpreter', 'latex');
        ylabel('ITD error [$\mu s$]', 'interpreter', 'latex');    
        head_rot_leg = cellstr(num2str(rad2deg(head_rot_az).'));
        head_rot_leg = strcat('Head rot =$',head_rot_leg,'^{\circ}$');    
        legend(head_rot_leg, 'interpreter', 'latex');
        set(gca, 'fontsize', 25, 'linewidth', 2);      

        %savefig
        if save_plot_flag
            mkdir(['plots/ITD_ILD_err/',arrTypeFold,'/headRot/']);
            export_fig(['plots/ITD_ILD_err/',arrTypeFold,'/headRot/ITD_err_Npw=',num2str(N_PW),'_',rigidArrTxt,'_Q=',num2str(Q_spiral),'_ra=',num2str(r_array),'_M=',num2str(M(m)),'_',LS_title(1:5),'.png'], '-transparent', '-r300');
        end
    end
    %}

end

%% Plots - average ILD
c_order = num2cell(get(gca,'colororder'), 2);
if COMPUTE_ILD
    % ILD error as func of M
    for h = 1:length(head_rot_az)
        figure('Position', [1, 1, 800, 400]);        
        % ILD subplot
        subplot(2, 1, 1);    
        plot(rad2deg(ph_DOA), squeeze(ILD_ref_av(:, h, 1)), 'linewidth', 3, 'color', c_order{1}); hold on; grid on;    
        plot(rad2deg(ph_DOA), squeeze(ILD_BSM_cmplx_av(:, h, :)), 'linewidth', 3, 'color', c_order{2});
        plot(rad2deg(ph_DOA), squeeze(ILD_BSM_mag_av(:, h, :)), 'linewidth', 3, 'color', c_order{3});
        xlim([0 ph_DOA_max]);
        ylabel('ILD$_{\mathrm{av}}$ [dB]', 'interpreter', 'latex');
        %title({[arr_txt, ' with head rotation = $',num2str(rad2deg(head_rot_az(h))),'^{\circ}$'],...
        %    ILD_fc_title}, 'interpreter', 'latex');
        %M_leg_cmplx = cellstr(num2str(M.'));
        %M_leg_cmplx = strcat('$M=',M_leg_cmplx,'$ cmplx');
        %M_leg_mag = cellstr(num2str(M.'));
        %M_leg_mag = strcat('$M=',M_leg_mag,'$ mag');
        %M_leg_all = [{'Ref'}; M_leg_cmplx; M_leg_mag];
        %legend(M_leg_all, 'interpreter', 'latex');
        leg_all = [{'Reference'}; {'BSM'}; {'BSM-MagLS'}];
        legend(leg_all, 'location', 'best', 'interpreter', 'latex');
        set(gca, 'fontsize', 20, 'fontname', 'times', 'linewidth', 2);  
        set(gca,'xticklabel',[]);

        % ILD error subplot
        subplot(2, 1, 2);
        plot(rad2deg(ph_DOA), squeeze(ILD_BSM_cmplx_av_err(:, h, :)), 'linewidth', 3, 'color', c_order{2}); hold on; grid on;    
        plot(rad2deg(ph_DOA), squeeze(ILD_BSM_mag_av_err(:, h, :)), 'linewidth', 3, 'color', c_order{3});
        xlim([0 ph_DOA_max]);
        ylim([0 15]);
        %xlabel('Azimuth [deg]', 'interpreter', 'latex');
        ylabel('$\epsilon_{\mathrm{ILD}_{\mathrm{av}}}$ [dB]', 'interpreter', 'latex');    
        %M_leg_cmplx = cellstr(num2str(M.'));
        %M_leg_cmplx = strcat('$M=',M_leg_cmplx,'$ cmplx');
        %M_leg_mag = cellstr(num2str(M.'));
        %M_leg_mag = strcat('$M=',M_leg_mag,'$ mag');
        %M_leg_all = [M_leg_cmplx; M_leg_mag];        
        %legend(M_leg_all, 'interpreter', 'latex');
        leg_all = [{'BSM'}; {'BSM-MagLS'}];
        legend(leg_all, 'location', 'best', 'interpreter', 'latex');
        set(gca, 'fontsize', 20, 'fontname', 'times', 'linewidth', 2);   
        
        % position
        ha=get(gcf,'children');
        %set(ha(1),'position',[.1 .5 1 0.5])
        set(ha(2),'position',[.1 .1 .87 .4])
        %set(ha(3),'position',[.1 .5 1 0.5])
        set(ha(4),'position',[.1 .55 .87 .4])

        %savefig
        if save_plot_flag
            %mkdir(['plots/ITD_ILD_err/',arrTypeFold,'/M/']);
            %export_fig(['plots/ITD_ILD_err/',arrTypeFold,'/M/ILD_err_Npw=',num2str(N_PW),'_',rigidArrTxt,'_Q=',num2str(Q_spiral),'_ra=',num2str(r_array),'_headRot=',num2str(rad2deg(head_rot_az(h)), '%d'),'_cmplxVSmag.png'], '-transparent', '-r300');
            
            export_fig(['/Users/liormadmoni/Google Drive/Lior/Acoustics lab/'...
                ,'Research/FB/Binaural_beamforming/Journal_paper/figs/ILD/'...
                ,'ILD_err_Npw=',num2str(N_PW),'_',sphereType,'_headRot=',num2str(rad2deg(head_rot_az(h)), '%d'),'_cmplxVSmag.png'], '-transparent', '-r300');
            
            
        end
    end

    % ILD error as func of head_rot
    %{
    for m = 1:length(M)

        figure('Position', [1, 1, 1280, 800]);        
        % ILD subplot
        subplot(2, 1, 1);    
        plot(rad2deg(ph_DOA), squeeze(ILD_ref_av(:, :, 1)), 'linewidth', 2); hold on; grid on;    
        plot(rad2deg(ph_DOA), squeeze(ILD_BSM_av(:, :, m)), 'linewidth', 2);
        xlim([0 ph_DOA_max]);
        ylabel('ILD [dB]', 'interpreter', 'latex');
        title({[arr_txt, ' with M = $',num2str(M(m)),'$'],...
            ILD_fc_title}, 'interpreter', 'latex');
        head_rot_leg = cellstr(num2str(rad2deg(head_rot_az).'));
        head_rot_leg1 = strcat('Ref head rot =$',head_rot_leg,'^{\circ}$');
        head_rot_leg2 = strcat('BSM Head rot =$',head_rot_leg,'^{\circ}$');
        head_rot_leg = [head_rot_leg1; head_rot_leg2];    
        legend(head_rot_leg, 'interpreter', 'latex');
        set(gca, 'fontsize', 25, 'linewidth', 2);    

        % ILD error subplot
        subplot(2, 1, 2);
        plot(rad2deg(ph_DOA), squeeze(ILD_av_err(:, :, m)), 'linewidth', 2); hold on; grid on;    
        xlim([0 ph_DOA_max]);
        xlabel('Azimuth [deg]', 'interpreter', 'latex');
        ylabel('ILD error [dB]', 'interpreter', 'latex');    
        head_rot_leg = cellstr(num2str(rad2deg(head_rot_az).'));
        head_rot_leg = strcat('Head rot =$',head_rot_leg,'^{\circ}$');
        legend(head_rot_leg, 'interpreter', 'latex');
        set(gca, 'fontsize', 25, 'linewidth', 2);    

        %savefig
        if save_plot_flag
            mkdir(['plots/ITD_ILD_err/',arrTypeFold,'/headRot/']);
            export_fig(['plots/ITD_ILD_err/',arrTypeFold,'/headRot/ILD_err_Npw=',num2str(N_PW),'_',rigidArrTxt,'_Q=',num2str(Q_spiral),'_ra=',num2str(r_array),'_M=',num2str(M(m)),'_',LS_title(1:5),'.png'], '-transparent', '-r300');
        end
    end
    %}

end


%% Plots - ILD as func of frequency
%{
cb_max_db = 10;
if COMPUTE_ILD
    f_plot = f_c_max_effective_freq(f_c_max_ERB_ind);
    % ILD error as func of M
    m = 1;
    for h = 1:length(head_rot_az)
        % cmplx LS
        figure('Position', [1, 1, 1280, 800]);                
        ILD_BSM_cmplx_tmp_err = squeeze(ILD_BSM_cmplx_freq_err(:, h, m, f_c_max_ERB_ind));
        surf(rad2deg(ph_DOA), f_plot, ILD_BSM_cmplx_tmp_err.',...
            'edgecolor', 'none'); view([0 90]);
        cb = colorbar; set(get(cb,'label'),'string','[dB]'); caxis([0 cb_max_db]);
        title(['Cmplx LS, M=',num2str(M(m)),', head-rot=',num2str(rad2deg(head_rot_az(h))),'$^{\circ}$'], 'interpreter', 'latex');
        xlabel('Azimuth [deg]', 'interpreter', 'latex');
        ylabel('ERB center frequency [Hz]', 'interpreter', 'latex');    
        xlim([0 ph_DOA_max]);
        ylim([f_plot(1) f_plot(end)]);
        set(gca, 'fontsize', 25, 'linewidth', 2);
        set(gca,'yscale','log')
        %savefig
        if save_plot_flag
            mkdir(['plots/ITD_ILD_err/',arrTypeFold,'/ILD_freq/']);
            export_fig(['plots/ITD_ILD_err/',arrTypeFold,'/ILD_freq/Npw=',num2str(N_PW),'_',sphereType,'_Q=',num2str(Q_spiral),'_ra=',num2str(r_array),'_M=',num2str(M(m)),'_headRot=',num2str(rad2deg(head_rot_az(h)), '%d'),'_cmplx.png'], '-transparent', '-r300');
        end
        
        % mag LS
        figure('Position', [1, 1, 1280, 800]);                
        ILD_BSM_cmplx_tmp_err = squeeze(ILD_BSM_mag_freq_err(:, h, m, f_c_max_ERB_ind));
        surf(rad2deg(ph_DOA), f_plot, ILD_BSM_cmplx_tmp_err.',...
            'edgecolor', 'none'); view([0 90]); 
        cb = colorbar; set(get(cb,'label'),'string','[dB]'); caxis([0 cb_max_db]);
        title(['Mag LS, M=',num2str(M(m)),', head-rot=',num2str(rad2deg(head_rot_az(h))),'$^{\circ}$'], 'interpreter', 'latex');
        xlabel('Azimuth [deg]', 'interpreter', 'latex');
        ylabel('ERB center frequency [Hz]', 'interpreter', 'latex');    
        xlim([0 ph_DOA_max]);              
        ylim([f_plot(1) f_plot(end)]);
        set(gca, 'fontsize', 25, 'linewidth', 2);    
        set(gca,'yscale','log')
        %savefig
        if save_plot_flag
            mkdir(['plots/ITD_ILD_err/',arrTypeFold,'/ILD_freq/']);
            export_fig(['plots/ITD_ILD_err/',arrTypeFold,'/ILD_freq/Npw=',num2str(N_PW),'_',sphereType,'_Q=',num2str(Q_spiral),'_ra=',num2str(r_array),'_M=',num2str(M(m)),'_headRot=',num2str(rad2deg(head_rot_az(h)), '%d'),'_mag.png'], '-transparent', '-r300');
        end
        
    end

end
%}








