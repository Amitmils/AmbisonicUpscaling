% Author: Tom Shlomo, ACLab BGU, 2020

% set path:
addpath(genpath('general'));

% set graphics defaults:
set(groot, 'defaultAxesTickLabelInterpreter','latex'); 
set(groot, 'defaultLegendInterpreter','latex');
set(groot,'defaulttextinterpreter','latex');
set(0, 'DefaultLineLineWidth', 2);
set(0, 'DefaultLineMarkerSize', 10);
set(0, 'defaultAxesXGrid', 'on')
set(0, 'defaultAxesYGrid', 'on')
set(0, 'defaultAxesZGrid', 'on')
set(groot,'defaultAxesXMinorGrid','on','defaultAxesXMinorGridMode','manual');
set(groot,'defaultAxesYMinorGrid','on','defaultAxesYMinorGridMode','manual');
set(groot,'defaultAxesZMinorGrid','on','defaultAxesZMinorGridMode','manual');
set(0,'DefaultTextFontSize',16) % Sets font size
set(0,'defaultAxesFontSize',16)
set(0,'DefaultAxesTitleFontSizeMultiplier', 1);