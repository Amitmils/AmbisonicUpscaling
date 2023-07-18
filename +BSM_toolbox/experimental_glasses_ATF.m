clearvars;
close all;
clc;

restoredefaultpath;
% add ACLtoolbox path
addpath(genpath('/Users/liormadmoni/Google Drive/Lior/Acoustics lab/Matlab/Research/Github/general'));
cd('/Users/liormadmoni/Google Drive/Lior/Acoustics lab/Matlab/Research/Github/general');

startup_script();
rng('default');

atf_fold = '/Users/liormadmoni/Google Drive/Lior/Acoustics lab/Matlab/Research/FB_BFBR/Data/Glasses_array/';

Vt = h5read(fullfile(atf_fold,'Device_ATFs.h5'),'/IR');
Vt = permute(Vt,[1,3,2]);
th_array = h5read(fullfile(atf_fold,'Device_ATFs.h5'),'/Theta');
ph_array = h5read(fullfile(atf_fold,'Device_ATFs.h5'),'/Phi');
originalFs = h5read(fullfile(atf_fold,'Device_ATFs.h5'),'/SamplingFreq_Hz');
% Vt = resample(Vt,fs,originalFs,'Dimension',2);