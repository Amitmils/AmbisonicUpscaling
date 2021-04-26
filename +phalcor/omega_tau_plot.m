function [h,cb] = omega_tau_plot(omega, tau, c, expected)

% Author: Tom Shlomo, ACLab BGU, 2020
arguments
    omega
    tau
    c
    expected
end
h=scatter3(omega(:,1)*180/pi, omega(:,2)*180/pi, tau*1e3, 100, c, '.');
cb=colorbar;
hold on;
if ~isempty(expected)
    plot3(expected.omega(:,1)*180/pi, expected.omega(:,2)*180/pi, expected.delay*1000, 'mo', 'MarkerSize', 20, "DisplayName", "Expected");
    if min(expected.delay)==0
        K = size(expected,1)-1;
        labels = ".  "+(0:K)';
    else
        K = size(expected,1);
        labels = ".  "+(1:K)';
    end
    text( expected.omega(:,1)*180/pi, expected.omega(:,2)*180/pi, expected.delay*1000, labels, "Color", "m", 'HorizontalAlignment', 'left');
end
% mylegend("Location", "SouthOutside", "Orientation", "Horizontal");
xlabel('$\theta$ [deg]');
ylabel('$\phi$ [deg]');
zlabel('$\tau$ [misec]');
xlim([0 180]);
ylim([-180 180]);

end