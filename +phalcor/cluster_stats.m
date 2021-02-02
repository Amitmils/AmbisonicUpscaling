function [C, expected] = cluster_stats(C, expected, tauTol, omegaTol)

% Author: Tom Shlomo, ACLab BGU, 2020

k = size(C,1);
tauErr = abs(C.tau - expected.delay');
omegaErr = angle_between(C.omega, expected.omega);
I_tau = tauErr <= tauTol;
I_omega = omegaErr <= omegaTol;
I = I_tau & I_omega;
C.isFA = ~any(I, 2);
C.cluster_id = (1:k)';

n = size(expected,1);
expected.clusterId = zeros(n,1);
expected.detected = false(n,1);
expected.omegaErr = inf(n,1);
expected.delayErr = inf(n,1);

for i=1:k
    if C.isFA(i)
        C.expId(i) = 0;
        C.tauErr(i) = nan;
        C.omegaErr(i) = nan;
    else
        II = find(I(i,:));
        [C.omegaErr(i), j] = min(omegaErr(i,II));
        j = II(j);
        C.tauErr(i) = tauErr(i,j);
        C.expId(i) = j;
        if expected.omegaErr(j) > C.omegaErr(i)
            expected.omegaErr(j) = C.omegaErr(i);
            expected.delayErr(j) = C.tauErr(i);
            expected.detected(j) = true;
            expected.clusterId(j) = i;
        end 
    end
end

I = expected.detected & expected.delay > 0;
fprintf("PD: %d/%d = %.2f\n", nnz(I), numel(I)-1, mean(expected.detected));
fprintf("FA: %d/%d = %.2f\n", nnz(C.isFA), size(C,1), mean(C.isFA));
fprintf("DOA RMS: %.1f deg (not including direct)\n", rms(expected.omegaErr(I))*180/pi);
fprintf("Delay RMS: %.1f usec (not including direct)\n", rms(expected.delayErr(I))*1e6);

end