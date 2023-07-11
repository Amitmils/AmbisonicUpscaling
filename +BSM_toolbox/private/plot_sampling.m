function plot_sampling(th, ph, markerSize, markerColor)
arguments
    th (:, :) double
    ph (:, :) double
    markerSize (1, 1) double = 16
    markerColor (1, 3) double = [0, 0, 0]
end
%PLOT_SAMPLING generates plots to illustrate the sampling points
% on a sphere.
% plot_sampling(th,ph);
% (th,ph) are the sampling points in spherical coordinates.
%
% Fundmentals of Spherical Array Processing
% Boaz Rafaely, 2018.

eps = 0.05;
[x,y,z] = s2c(th,ph,1 + eps);

AxisFontSize=16;

figure;
n=48;
[X,Y,Z] = sphere(n);
h=surf(X,Y,Z);
colormap(bone);
set(h,'EdgeAlpha',0.1);
axis tight equal
axis off
hold on;
v=plot3(x,y,z,'.');
set(v,'MarkerSize',markerSize, 'MarkerEdgeColor', markerColor,'MarkerFaceColor','k');
%export_fig test.png -transparent -r300;

% figure;
% v=plot((180/pi)*ph,(180/pi)*th,'.','Color',[0 0 0.5]);
% set(gca,'FontSize',AxisFontSize);
% set(v,'MarkerSize',AxisFontSize);
% xlabel('$\phi$ (degrees)','FontSize',AxisFontSize,'Interp','Latex');
% ylabel('$\theta$ (degrees)','FontSize',AxisFontSize,'Interp','Latex');
% axis([0,360,0,180]);
