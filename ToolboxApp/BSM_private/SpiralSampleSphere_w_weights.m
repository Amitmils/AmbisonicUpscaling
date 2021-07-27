function [a_spiral, th_spiral, ph_spiral] = SpiralSampleSphere_w_weights(Q)

% calculate spiral sampling on sphere
[V, ~] = SpiralSampleSphere(Q);

% convert to spherical coordiantes (Boaz notation)
[th_spiral, ph_spiral, ~] = c2s(V(:, 1), V(:, 2), V(:, 3));
% ph_spiral(ph_spiral < 0) = ph_spiral(ph_spiral < 0) + 2*pi;
ph_spiral = ph_spiral + pi;
% plot_sampling(theta_ps.', ph_spiral.');

N_est = min([1, floor(sqrt(Q) - 1)]);

% estimate weights
a_spiral = real( EstimateIntegrationWeights(N_est, th_spiral, ph_spiral, 'diagonal') );

end