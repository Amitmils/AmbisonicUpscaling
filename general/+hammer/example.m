% Author: Tom Shlomo, ACLab BGU, 2020

N = 10; % SH order
K = 4; % num of waves
omega = rand_on_sphere(K); % DOAs
x = randn(K,1)+1i*randn(K,1); % amplitudes
Y = shmat(N, omega); % SH matrix
anm = Y'*x;

hammer.surf([], anm);
hammer.plot(omega(:,1), omega(:,2), 'rx');
% you could also use:
% hammer.plot(omega, [], 'rx')
% instead

% hold on is not required
