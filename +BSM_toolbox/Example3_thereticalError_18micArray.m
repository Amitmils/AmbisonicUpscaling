%% This script calculates the theoretical error of BSM with the 18-mic head array

% Date created: June 20, 2021
% Created by:   Lior Madmoni

clearvars;
close all;
clc;

restoredefaultpath;
% add ACLtoolbox path
addpath(genpath('/Users/liormadmoni/Google Drive/Lior/Acoustics lab/Matlab/Research/Github/general'));
cd('/Users/liormadmoni/Google Drive/Lior/Acoustics lab/Matlab/Research/Github/general/');

startup_script();
rng('default');

% parameters/flags - array
filt_len = 0.032; %0.032;                              % filters (BSM/HRTF) length [sec]
arrayType = 1;                                         % 0 - 90 equiangle horizontal plane 1m, 1 - 110 equiangle horizontal plane 3m
head_rot_az = ...
    wrapTo2Pi(deg2rad([0 45 90]));                         % vector of head rotations [rad]
normSV = false;                                         % true - normalize steering vectors

% parameters/flags - general
c = 343;                                               % speed of sound [m/s]
desired_fs = 48000;                                    % choose samplong frequency in Hz
saveFiles = false;                                     % save MATLAB files before time interpolation?
save_plot_flag = false;

% parameters/flags - BSM design
inv_opt = 1;                                           % opt=1 -> ( (1 / lambda) * (A * A') + eye )  |||| opt2=1 -> ((A * A') + lambda * eye);

magLS = true;                                          % true - magLS, false - complex LS
f_cut_magLS = 1500;                                    % above cutoff frequency use MagLS
tol_magLS = 1e-20;    
max_iter_magLS = 1E5;
%noise
SNR = 20;                                              % assumed sensors SNR [dB]
sig_n = 0.1;
sig_s = 10^(SNR/10) * sig_n;
SNR_lin = sig_s / sig_n;    
%signal
filt_samp     = filt_len * desired_fs;
freqs_sig    = ( 0 : (filt_samp / 2) ) * desired_fs / filt_samp;
freqs_sig(1) = 1/4 * freqs_sig(2); %to not divide by zero

% Text variables for plots 
switch arrayType 
    case 0
        arrayTypeTxt = ['90 equiangle horizontal plane 1m'];
        arrayTypeLbl = ['18mic_90equiangle_1m'];
    case 1
        arrayTypeTxt = ['110 equiangle horizontal plane 3m'];
        arrayTypeLbl = ['18mic_110equiangle_3m'];
end
if magLS    
    %LS_title = ['MagLS max-iter=',num2str(max_iter_number_magLS, '%1.1e'),' tol=',num2str(tol_magLS,'%1.1e')];
    LS_title = ['MagLS_fcut=',num2str(f_cut_magLS)];
else
    LS_title = 'RegLS';
end

%% ================= Load Array Data
arrayFold = '/Users/liormadmoni/Google Drive/Lior/Acoustics lab/Matlab/Research/FB_BFBR/Data/18mic_head_array/data_mesh2hrtf/';
switch arrayType
    case 0
        arrayData = load([arrayFold,'head_irs_circle_1m.mat']);
    case 1
        arrayData = load([arrayFold,'head_irs_circle_3m.mat']);
end
M = size(arrayData.irs, 2);

%resample IRs
if desired_fs ~= arrayData.fs
    arrayData.irs = resample(arrayData.irs, desired_fs, arrayData.fs);
    arrayData.fs = desired_fs;
end

% interpolate IRs to length fo filt_samp
arrayData.TFs = fft(arrayData.irs, filt_samp, 1);
%trim negative frequencies
arrayData.TFs = arrayData.TFs(1:ceil(filt_samp/2)+1, :, :);

th_BSMgrid_vec = arrayData.colatitude;
ph_BSMgrid_vec = arrayData.azimuth;

%% ================= HRTFS preprocessing
% load HRIRs
N_HRTF = 30;
HRTFpath =  '/Users/liormadmoni/Google Drive/ACLtoolbox/Data/HRTF/earoHRIR_KU100_Measured_2702Lebedev.mat';
load(HRTFpath);         % hobj is HRIR earo object
hobj.shutUp = false;
%%Interpolate HRTF to frequencies
hobj_freq_grid = hobj;
if strcmp(hobj_freq_grid.dataDomain{1},'FREQ'), hobj_freq_grid=hobj_freq_grid.toTime(); end
% resample HRTF to desired_fs
hobj_freq_grid = hobj_freq_grid.resampleData(desired_fs);
hobj_freq_grid = hobj_freq_grid.toFreq(filt_samp);
% Trim negative frequencies
hobj_freq_grid.data = hobj_freq_grid.data(:, 1:ceil(filt_samp/2)+1, :);

%% ================= Load WignerD Matrix
WignerDpath = '/Users/liormadmoni/Google Drive/Lior/Acoustics lab/Matlab/Research/FB_BFBR/Data/WignerDMatrix_diagN=32.mat';
load(WignerDpath);
N_HRTF_rot = 30;
DN = (N_HRTF_rot + 1)^2; % size of the wignerD matrix
D_allAngles = D(:, 1 : DN);

%% Create BSM object to pass parameters to function more easily
BSMobj.freqs_sig = freqs_sig;
BSMobj.magLS = magLS;
BSMobj.f_cut_magLS = f_cut_magLS;
BSMobj.tol_magLS = tol_magLS;
BSMobj.max_iter_magLS = max_iter_magLS;
BSMobj.c = c;
BSMobj.th_BSMgrid_vec = th_BSMgrid_vec;
BSMobj.ph_BSMgrid_vec = ph_BSMgrid_vec;
BSMobj.normSV = normSV;
BSMobj.SNR_lin = SNR_lin;
BSMobj.inv_opt = inv_opt;
BSMobj.head_rot_az = head_rot_az;
BSMobj.M = M;
BSMobj.desired_fs = desired_fs;
BSMobj.filt_samp = filt_samp;
%%=Update BSM object 
BSMobj.n_mic = M;                
BSMobj.th_array = arrayData.colatitude;
BSMobj.ph_array = arrayData.azimuth;

%% ================= reshape array steering vectors according to BSM convention [n_mic x Q x freq]
V_k = permute(arrayData.TFs, [2 3 1]);    

%% ================= BSM filters calculation
err_pl = zeros(length(freqs_sig), length(head_rot_az));
err_pr = zeros(length(freqs_sig), length(head_rot_az));

for h = 1 : length(head_rot_az)    
    tic;    
    %% ================= Rotate HRTFs according to head rotation - new        
    hobj_rot = RotateHRTF(hobj_freq_grid, N_HRTF_rot, D_allAngles, head_rot_az(h));
    % Interpolate HRTF to array grid
    hobj_rot_BSM = hobj_rot;
    hobj_rot_BSM = hobj_rot_BSM.toSpace('SRC', th_BSMgrid_vec, ph_BSMgrid_vec);  

    %% ================= Generate BSM filters in frequency domain    
    [c_BSM_comp_l, c_BSM_comp_r] = BSM_toolbox.GenerateBSMfilters_faster(BSMobj, V_k, hobj_rot_BSM);
    
    %% ================= error calculation
    for f = 1:length(freqs_sig)
        % HRTFs
        h_l = hobj_rot_BSM.data(:, f, 1);
        h_r = hobj_rot_BSM.data(:, f, 2);
        
        if ~magLS
            % error for complex LS
            err_pl(f, h) = sig_s * norm(V_k(:, :, f)' * c_BSM_comp_l(:, f) - conj(h_l), 2)^2 + sig_n * norm(c_BSM_comp_l(:, f), 2)^2;
            err_pl(f, h) = err_pl(f, h) / (sig_s * norm(conj(h_l), 2)^2);
            err_pl(f, h) = sqrt(err_pl(f, h));
            err_pr(f, h) = sig_s * norm(V_k(:, :, f)' * c_BSM_comp_r(:, f) - conj(h_r), 2)^2 + sig_n * norm(c_BSM_comp_r(:, f), 2)^2;
            err_pr(f, h) = err_pr(f, h) / (sig_s * norm(conj(h_r), 2)^2);
            err_pr(f, h) = sqrt(err_pr(f, h));
        else
            % error for mag LS
            err_pl(f, h) = sig_s * norm(abs(V_k(:, :, f)' * c_BSM_comp_l(:, f)) - abs(h_l), 2)^2 + sig_n * norm(c_BSM_comp_l(:, f), 2)^2;
            err_pl(f, h) = err_pl(f, h) / (sig_s * norm(conj(h_l), 2)^2);
            err_pl(f, h) = sqrt(err_pl(f, h));
            err_pr(f, h) = sig_s * norm(abs(V_k(:, :, f)' * c_BSM_comp_r(:, f)) - abs(h_r), 2)^2 + sig_n * norm(c_BSM_comp_r(:, f), 2)^2;
            err_pr(f, h) = err_pr(f, h) / (sig_s * norm(conj(h_r), 2)^2);
            err_pr(f, h) = sqrt(err_pr(f, h));
        end
    end

    % print status    
    if h == 1
        t1 = toc;
        time_now = datetime('now');
        time_to_finish = time_now + seconds(t1 * (length(head_rot_az) - 1));    
        FinishTimeString = datestr(time_to_finish);
        
        fprintf('Time for one loop over head rotation is %.2f\n', t1);
        fprintf(['Estimated finish time is ',FinishTimeString,'\n\n']);
        fprintf('Finished BSM reproduction for head rotation index %d/%d', h, length(head_rot_az));
    else
        fprintf(repmat('\b', 1, (length(num2str(length(head_rot_az))) + 1 + (length(num2str(h - 1))))));
        fprintf('%d/%d',h, length(head_rot_az));
    end
end
fprintf('\n');

%% Plots
plt_colors = get(gca, 'colororder');
plt_linwid = {5; 5; 5};
plt_marker = {'+' ; 'o' ; '*'};    % { '+' ; 'o' ; '*'; '.'; 'x'; 'square'; 'diamond'; '^'; '>'; '<'; 'pentagram'}
plt_lbls = true;

for h = 1:length(head_rot_az)

    % error in binaural signal estimation as function of frequency
    err_pl_dB = mag2db(err_pl(:, h));
    err_pr_dB = mag2db(err_pr(:, h));
%     figure('Position', [1, 1, 1280, 800]);
    figure('Position', [1, 1, 800, 400]);
    sp2 = semilogx(freqs_sig, err_pl_dB, 'linewidth', 7, 'DisplayName', 'left');
    hold on;
    sp2 = semilogx(freqs_sig, err_pr_dB, 'linewidth', 7, 'DisplayName', 'right');
%     set(sp2, {'Marker'} , plt_marker );
    grid on;
    axis tight;
    if plt_lbls
        title({['Head rotation = ',num2str(rad2deg(head_rot_az(h))),'$^{\circ}$']},...
            {arrayTypeTxt}, 'interpreter', 'latex');

        xlabel('Frequency (Hz)', 'interpreter', 'latex');
        ylabel('$|| \hat{p}^{l,r}(k) - p^{l,r}(k) ||_2 / || p^{l,r}(k) ||_2$ (dB)', 'interpreter', 'latex');
    end
    legend('location', 'east', 'interpreter', 'latex');
    set(gca, 'fontsize', 20, 'linewidth', 2, 'fontname', 'times');
    xlim([75 10000]);

    %savefig
    if save_plot_flag
        export_fig(['/Users/liormadmoni/Google Drive/Lior/Acoustics lab/Matlab/Research/FB_BFBR/BSM/plots/err_pl_',arrayTypeLbl,'_arrayRot=',num2str(rad2deg(head_rot_az(h))),'.png'], '-transparent', '-r300');
    end
end


clear c_BSM_comp_l_tmp_time_cs c_BSM_comp_r_tmp_time_cs c_BSM_NoComp_l_tmp_time_cs c_BSM_NoComp_r_tmp_time_cs
clear c_BSM_comp_l_tmp c_BSM_comp_r_tmp c_BSM_NoComp_l_tmp c_BSM_NoComp_r_tmp
% BSMobj = rmfield(BSMobj,{'n_mic', 'th_array', 'ph_array'});

%% Save mat files
%{
if saveFiles
    time_now = datetime('now');
    DateString = datestr(time_now, 30);
    mkdir(['/+BSM_toolbox/Data/']);
    
    % save only BSM output
    save(['/+BSM_toolbox/Data/BSMfilter_',arrayTypeTxt,'_N_PW=',num2str(N_PW),'_',LS_title,'_',DateString,'.mat'],...
        'c_BSM_comp_l', 'c_BSM_comp_r', 'c_BSM_NoComp_l', 'c_BSM_NoComp_r',...        
        'BSMobj', 'arrayTypeTxt');
end
%}






