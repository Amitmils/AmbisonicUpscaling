%% This script generates binaural signals from pre-calculated BSM filters 

% Date created: July 19, 2022
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

%% === load array recordings
recording_path = '/Volumes/GoogleDrive/My Drive/Lior/Acoustics lab/Research/FB/Binaural_beamforming/Zamir problems/dataToShare/Capture examples/';
recording_name = 'Recordings_allMics_excerpt02.wav'; % Recordings_allMics_excerpt01.wav / Recordings_allMics_excerpt02.wav
[x_mic, x_fs] = audioread([recording_path,recording_name]);
n_mic = size(x_mic, 2);


%% === load BSM filters
filters_path = '/Volumes/GoogleDrive/My Drive/Lior/Acoustics lab/Research/FB/Binaural_beamforming/Zamir problems/Lior analysis/';
filters_name = 'BSMfilters_20220701T095959.mat';
load([filters_path, filters_name]);


%% === generate BSM signals

% choose filters
c_l = c_BSM_CVX_l_time_cs; % c_BSM_CVX_l_time_cs / c_BSM_VEM_l_time_cs
c_r = c_BSM_CVX_r_time_cs; % c_BSM_CVX_r_time_cs / c_BSM_VEM_r_time_cs

% Time reversal - to conjugate filters in frequency domain        
c_l = [c_l(:, 1), c_l(:, end:-1:2)];
c_r = [c_r(:, 1), c_r(:, end:-1:2)];

% zero-pad array recording to correct length
filt_samp = size(c_l, 2);
p_array_t = x_mic;
p_array_t_zp = [p_array_t; zeros(filt_samp - 1, n_mic)];

% filter with fftfilt
p_BSM_t_l = (sum(fftfilt(c_l.', p_array_t_zp), 2));
p_BSM_t_r = (sum(fftfilt(c_r.', p_array_t_zp), 2));                            

%% Listen to results
p_BSM_t = cat(2, p_BSM_t_l, p_BSM_t_r);

% normalize 
p_BSM_t = 0.9 * (p_BSM_t ./ max(max(p_BSM_t)));

%soundsc(p_BSM_t, desired_fs);

%%
%
output_path = sprintf([filters_path, 'signals/', 'BSM_CVX.wav']);
save_audio(output_path, p_BSM_t, x_fs);
%}


%% Utility functions
function save_audio(file_path, p, fs)
    [folder_path, ~, ~] = fileparts(file_path);
    if ~exist(folder_path, 'dir')
       mkdir(folder_path)
    end
    p = 0.9 * p / max(max(abs(p)));
    audiowrite(file_path, p, fs);
end



