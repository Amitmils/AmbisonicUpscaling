function hFigureHandle = roomSimulationPlot_ISF(RoomDims, srcPos, recPos, srcView)
% Function to plot room simulation setup
% 
% INTPUTS:
%    RoomDims: Room dimension in meters [x y z]
%    srcPos : source position(s) in radius [m], elevation [rad], azimuth [rad],
%               relative to the receiver position RecPos
%    recPos : x/y/z position of the receiver [m]
%    srcView :  Azimuth and elevation orientation of the source(s) following the
%               coordinate convention from rs.RecAz, and rs.RecEl
%               []                      : sources are oriented towards the receiver
%               [az1 el1; ..., azN elN] : sources are oriented according to specified
%                                          values (one line per source)
%
% NOTES:
%       Written by Zamir Ben-Hur 22.8.18 (based on code from AKtools)
%

% --- check if the source view was passed --- %
if ~exist('srcView','var')
    tmp = [srcPos(:,3) pi/2 - srcPos(:,2)]*180/pi;
    
    % wrap to 360 deg
    tmp(:,1) = mod(tmp(:,1), 360);
    
    % mirror to get source rotation
    srcView = [mod(180+tmp(:,1), 360) -tmp(:,2)];
    clear tmp
    
end

% viewing direction of the receiver
% Azimuth:
%      90: rec. points to positive y direction
%       0: rec. points to positive x direction
% -90/270: rec. points to negative y direction
% Elevation:
%      90: rec. points to positive z direction
%     -90: rec. points to negative z direction
recView = [0 0];

% --- get the source position(s) in carthesian coordinates --- %
[x, y, z] = sph2cart(srcPos(:,3), pi/2 - srcPos(:,2), srcPos(:,1));
srcPos = [x + recPos(1) y + recPos(2) z + recPos(3)];


clear x y z


% create figure
figureSize = [20 15];

hFigureHandle = figure;
%%%%%%%%%%%%%%%%%%%%%% add this line if you don't want to see the room
%set(hFigureHandle,'Visible', 'off');
%%%%%%%%%%%%%%%%%%%%%%

% get screensize in centimeter
pixpercm = get(0,'ScreenPixelsPerInch')/2.54;
screen = get(0,'ScreenSize')/pixpercm;
% get position for figure
left   = max([(screen(3)-figureSize(1))/2 0]);
bottom = max([(screen(4)-figureSize(2))/2 0]);

set(hFigureHandle,'PaperUnits', 'centimeters');
set(hFigureHandle,'Units', 'centimeters');

% paper size for printing
set(hFigureHandle, 'PaperSize', figureSize);
% location on printed paper
set(hFigureHandle,'PaperPosition', [.1 .1 figureSize-.1]);
% location and size on screen
set(hFigureHandle,'Position', [left bottom figureSize]);

% set color
set(hFigureHandle, 'color', [1 1 1])

% plot
hold on
% -------- plot the room -------- %

% Room width
width = min(RoomDims(1:2)) ; 

% Regular mesh of points on the floor
stp = max(.25,(2^nextpow2(width/4))/4) ;
xFlr = (0:stp:RoomDims(1))' ;
yFlr = (0:stp:RoomDims(2))' ;
if RoomDims(1) > max(xFlr)
    xFlr = [ xFlr ; RoomDims(1) ] ;
end
if RoomDims(2) > max(yFlr)
    yFlr = [ yFlr ; RoomDims(2) ] ;
end

% Plot a checkered floor
for I = 1 : length(xFlr)-1
    for J = 1 : length(yFlr)-1
        % Color of the (I,J) tile
        col = [.7 .7 .7] + .1 * rem(I+J,2) ;
        % Create a patch corresponding to the (I,J) tile
        patch([xFlr(I) xFlr(I+1) xFlr(I+1) xFlr(I)], ...
              [yFlr(J) yFlr(J) yFlr(J+1) yFlr(J+1)], ...
              [0 0 0 0],'facecolor',col,'edgecolor','none') ;
    end
end

% -------- plot receiver and source(s) -------- %
% get a sphere for marking the positions
r = min(1, .05 * min(RoomDims));
[xS, yS, zS] = sphere(10);
xS = xS * r;
yS = yS * r;
zS = zS * r;

% plot the receiver position
surf(xS+recPos(1), yS+recPos(2), zS+recPos(3), 'FaceColor', AKcolors('r'), 'EdgeColor', 'none')

% plot the receiver orientation
recAz = recView(1);
recEl = recView(2);
[x, y, z] = sph2cart(mean(recAz)/180*pi, mean(recEl)/180*pi, 1.3*r);
plot3([recPos(1) x+recPos(1)], [recPos(2) y+recPos(2)], [recPos(3) z+recPos(3)], 'k', 'LineWidth', 4)

% plot the source position(s)
sources = 1:size(srcPos, 1);
for nn = 1:numel(sources)
    
    x = srcPos(sources(nn),1);
    y = srcPos(sources(nn),2);
    z = srcPos(sources(nn),3);
    
    % plot the source
    if (nn == 1)
        surf(xS+x, yS+y, zS+z, 'FaceColor', AKcolors('b'), 'EdgeColor', 'none')
    else
        surf(xS+x, yS+y, zS+z, 'FaceColor', AKcolors('o'), 'EdgeColor', 'none')
    end
    %surf(xS+x, yS+y, zS+z, 'FaceColor', AKcolors('b'), 'EdgeColor', 'none')
    
    % plot the source orientation
    srcAz = srcView(sources(nn),1);
    srcEl = srcView(sources(nn),2);
    [xR, yR, zR] = sph2cart(srcAz/180*pi, srcEl/180*pi, 1.3*r);
    plot3([x x+xR], [y y+yR], [z z+zR], 'k', 'LineWidth', 4)
    
end

% -------- format the plot -------- %

% Axes
axis equal
axis([0 RoomDims(1) 0 RoomDims(2) 0 RoomDims(3)])
axis vis3d
box on

% Perspective
camproj perspective
campos([-2 -7 10].*RoomDims)

% Lighting
light('Position',[0.5 0.5 0]+[0 0 1.5].*RoomDims,'Style','local')
lighting phong
material dull

% Axis labels
xlabel('x [m]')
ylabel('y [m]')
zlabel('z [m]')


title({'Scene geometry (source = blue, receiver = red)' 'Black dots indicate viewing direction'})
set(gca,'FontSize',14)

if ~nargout
    clearvars hFigureHandle
end

end