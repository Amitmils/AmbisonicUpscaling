%% This script calculates the theoretical error of BSM with the spherical/circular/semi-circular array

% Date created: June 21, 2021
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
filt_len = 0.032; %0.032;                                % filters (BSM/HRTF) length [sec]
arrayType = 1;                                         % 0 - spherical array, 1 - semi-circular array, 2 - full-circular array
rigidArray = 1;                                        % 0 - open array, 1 - rigid array
M = 6;                                                 % number of microphones
r_array = 0.1;                                         % array radius
array_rot_az = ...
    wrapTo2Pi(deg2rad([0 45 90]));                     % vector of head rotations [degrees]
normSV = true;                                         % true - normalize steering vectors

% parameters/flags - general
c = 343;                                               % speed of sound [m/s]
desired_fs = 48000;                                    % choose samplong frequency in Hz
N_PW = 30;                                             % SH order of plane-wave synthesis
save_plot_flag = false;                                     % save MATLAB files before time interpolation?

% parameters/flags - BSM design
inv_opt = 1;                                           % opt=1: ( (1 / lambda) * (A * A') + eye ) | opt=2: ((A * A') + lambda * eye);
source_distribution = 1;                               % 0 - nearly uniform (t-design), 1 - spiral nearly uniform
Q = 240;                                               % Assumed number of sources

magLS = false;                                          % true - magLS, false - complex LS
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
if magLS    
    %LS_title = ['MagLS max-iter=',num2str(max_iter_number_magLS, '%1.1e'),' tol=',num2str(tol_magLS,'%1.1e')];
    LS_title = ['MagLS_fcut=',num2str(f_cut_magLS)];
else
    LS_title = 'RegLS';
end

%% ================= HRTFS preprocessing
% load HRIRs
N_HRTF = 30;
HRTFpath =  '/Users/liormadmoni/Google Drive/ACLtoolbox/Data/HRTF/earoHRIR_KU100_Measured_2702Lebedev.mat';
load(HRTFpath);         % hobj is HRIR earo object
hobj.shutUp = false;
[th_BSMgrid_vec, ph_BSMgrid_vec] = BSM_toolbox.BSMgrid(source_distribution, Q);
hobj = BSM_toolbox.HRTF2BSMgrid(hobj, N_HRTF, th_BSMgrid_vec, ph_BSMgrid_vec);
%%Interpolate HRTF to frequencies
hobj_freq_grid = hobj;
if strcmp(hobj_freq_grid.dataDomain{1},'FREQ'), hobj_freq_grid=hobj_freq_grid.toTime(); end
% resample HRTF to desired_fs
hobj_freq_grid = hobj_freq_grid.resampleData(desired_fs);
hobj_freq_grid = hobj_freq_grid.toFreq(filt_samp);
% Trim negative frequencies
hobj_freq_grid.data = hobj_freq_grid.data(:, 1:ceil(filt_samp/2)+1, :);

%% Create BSM object to pass parameters to function more easily
BSMobj.r_array = r_array;
BSMobj.N_PW = N_PW;
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
BSMobj.rigidArray = rigidArray;
BSMobj.array_rot_az = array_rot_az;
BSMobj.M = M;
BSMobj.Q = Q;
BSMobj.source_distribution = source_distribution;
BSMobj.desired_fs = desired_fs;
BSMobj.filt_samp = filt_samp;
BSMobj.sphereType = sphereType;

%% ================= BSM filters calculation
% ===== initialization
err_pl = zeros(length(freqs_sig), length(array_rot_az));
err_pr = zeros(length(freqs_sig), length(array_rot_az));

for ar = 1 : length(array_rot_az)    
    tic;
    for n = 1 : length(M)            
        %% ================= Get array positions
        n_mic = M(n);        
        [th_array, ph_array, ph_rot_array] = BSM_toolbox.GetArrayPositions(arrayType, n_mic, array_rot_az(ar));
        
        %% Update BSM object 
        BSMobj.n_mic = n_mic;                
        BSMobj.th_array = th_array;
        %no compensation
%         BSMobj.ph_array = ph_array;
        %with compensation
        BSMobj.ph_array = ph_rot_array;
        
        %% ================= calculate array steering vectors (for BSM filters)        
        N_SV = N_PW;        
        V_k = CalculateSteeringVectors(BSMobj, N_SV, th_BSMgrid_vec, ph_BSMgrid_vec); 
        V_k = permute(V_k, [3 2 1]);
        
        %% ================= Generate BSM filters in frequency domain
        % Non-compensated version of BSM - use original array orientation
%         [c_BSM_NoComp_l_tmp, c_BSM_NoComp_r_tmp] = BSM_toolbox.GenerateBSMfilters_faster(BSMobj, V_k, hobj_freq_grid);
        % The compensated version of BSM - use rotated array orientation        
        [c_BSM_comp_l, c_BSM_comp_r] = BSM_toolbox.GenerateBSMfilters_faster(BSMobj, V_k, hobj_freq_grid);
        
        %% ================= error calculation
        for f = 1:length(freqs_sig)
            % HRTFs
            h_l = hobj_freq_grid.data(:, f, 1);
            h_r = hobj_freq_grid.data(:, f, 2);

            if ~magLS
                % error for complex LS
                err_pl(f, ar) = sig_s * norm(V_k(:, :, f)' * c_BSM_comp_l(:, f) - conj(h_l), 2)^2 + sig_n * norm(c_BSM_comp_l(:, f), 2)^2;
                err_pl(f, ar) = err_pl(f, ar) / (sig_s * norm(conj(h_l), 2)^2);
                err_pl(f, ar) = sqrt(err_pl(f, ar));
                err_pr(f, ar) = sig_s * norm(V_k(:, :, f)' * c_BSM_comp_r(:, f) - conj(h_r), 2)^2 + sig_n * norm(c_BSM_comp_r(:, f), 2)^2;
                err_pr(f, ar) = err_pr(f, ar) / (sig_s * norm(conj(h_r), 2)^2);
                err_pr(f, ar) = sqrt(err_pr(f, ar));
            else
                % error for mag LS
                err_pl(f, ar) = sig_s * norm(abs(V_k(:, :, f)' * c_BSM_comp_l(:, f)) - abs(h_l), 2)^2 + sig_n * norm(c_BSM_comp_l(:, f), 2)^2;
                err_pl(f, ar) = err_pl(f, ar) / (sig_s * norm(conj(h_l), 2)^2);
                err_pl(f, ar) = sqrt(err_pl(f, ar));
                err_pr(f, ar) = sig_s * norm(abs(V_k(:, :, f)' * c_BSM_comp_r(:, f)) - abs(h_r), 2)^2 + sig_n * norm(c_BSM_comp_r(:, f), 2)^2;
                err_pr(f, ar) = err_pr(f, ar) / (sig_s * norm(conj(h_r), 2)^2);
                err_pr(f, ar) = sqrt(err_pr(f, ar));
            end
        end
    
    end
            
    % print status    
    if ar == 1
        t1 = toc;
        time_now = datetime('now');
        time_to_finish = time_now + seconds(t1 * (length(array_rot_az) - 1));    
        FinishTimeString = datestr(time_to_finish);
        
        fprintf('Time for one loop over head rotation is %.2f\n', t1);
        fprintf(['Estimated finish time is ',FinishTimeString,'\n\n']);
        fprintf('Finished BSM reproduction for head rotation index %d/%d', ar, length(array_rot_az));
    else
        fprintf(repmat('\b', 1, (length(num2str(length(array_rot_az))) + 1 + (length(num2str(ar - 1))))));
        fprintf('%d/%d',ar, length(array_rot_az));
    end
end
fprintf('\n');

%% Plots
plt_colors = get(gca, 'colororder');
plt_linwid = {5; 5; 5};
plt_marker = {'+' ; 'o' ; '*'};    % { '+' ; 'o' ; '*'; '.'; 'x'; 'square'; 'diamond'; '^'; '>'; '<'; 'pentagram'}
plt_lbls = true;

for ar = 1:length(array_rot_az)

    % error in binaural signal estimation as function of frequency
    err_pl_dB = mag2db(err_pl(:, ar));
    err_pr_dB = mag2db(err_pr(:, ar));
%     figure('Position', [1, 1, 1280, 800]);
    figure('Position', [1, 1, 800, 400]);
    sp2 = semilogx(freqs_sig, err_pl_dB, 'linewidth', 7, 'DisplayName', 'left');
    hold on;
    sp2 = semilogx(freqs_sig, err_pr_dB, 'linewidth', 7, 'DisplayName', 'right');
%     set(sp2, {'Marker'} , plt_marker );
    grid on;
    axis tight;
    if plt_lbls
        title({['Head rotation = ',num2str(rad2deg(array_rot_az(ar))),'$^{\circ}$']},...
            {arrayTypeTxt}, 'interpreter', 'latex');

        xlabel('Frequency (Hz)', 'interpreter', 'latex');
        ylabel('$|| \hat{p}^{l,r}(k) - p^{l,r}(k) ||_2 / || p^{l,r}(k) ||_2$ (dB)', 'interpreter', 'latex');
    end
    legend('location', 'east', 'interpreter', 'latex');
    set(gca, 'fontsize', 20, 'linewidth', 2, 'fontname', 'times');
    xlim([75 10000]);

    %savefig
    if save_plot_flag
        export_fig(['/Users/liormadmoni/Google Drive/Lior/Acoustics lab/Matlab/Research/FB_BFBR/BSM/plots/err_pl_',arrayTypeTxt,'_arrayRot=',num2str(rad2deg(array_rot_az(ar))),'.png'], '-transparent', '-r300');
    end
end

%{
clear c_BSM_comp_l_tmp_time_cs c_BSM_comp_r_tmp_time_cs c_BSM_NoComp_l_tmp_time_cs c_BSM_NoComp_r_tmp_time_cs
clear c_BSM_comp_l_tmp c_BSM_comp_r_tmp c_BSM_NoComp_l_tmp c_BSM_NoComp_r_tmp
BSMobj = rmfield(BSMobj,{'n_mic', 'th_array', 'ph_array'});
%}





