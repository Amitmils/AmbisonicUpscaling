function [C, expected2] = fine_tune_delay(C, T3, Jf, delta_f, opts)
arguments
    C table
    T3 table
    Jf (1,1) double
    delta_f (1,1) double
    opts.expected table
    opts.plot_flag (1,1) logical = 0
    opts.real_flag (1,1) logical = 1
    opts.max_tau_err = 300e-6;
    opts.tau_fine_res = 0.1e-6;
    opts.init_tau_res = 1e-6;
    opts.grid_size = 100;
    opts.verbose = 1;
    opts.omegaTol = 15*pi/180;
    opts.tauTol = 0.5e-3;
end

% Author: Tom Shlomo, ACLab BGU, 2020

N = sqrt(size(T3.u, 2))-1;
v_dir = shmat(N, C.omega(C.expId==1, :))';
for c=1:size(C,1)
    if C.tau(c) == 0
        continue
    end
    tau_hat_prime = C.tau(c);
    initial_error = C.tauErr(c);
    
    v_ref = shmat(N, C.omega(c,:))';
    A = T3(T3.c == C.cluster_id(c), :);
    if ~C.isFA(c)
        C.tau_exp(c) = opts.expected.delay(opts.expected.clusterId == C.cluster_id(c));
    else
        C.tau_exp(c) = nan;
    end
    
    M = size(A,1);
    A.beta = zeros(M,1);
    for n=1:M
        A.beta(n) = A.sigma(n) * (v_ref' * A.u(n,:).') * (conj(A.v(n,:)) * v_dir);
    end
    
    f_c = A.f + (delta_f)*(Jf-1)/2;
    func1 = @(tau) exp(-1i*2*pi*(f_c.').*(A.tau' - tau(:)))*A.beta;
    
    if opts.real_flag
        tau_res = 0.01/sqrt(mean(f_c.^2));
        func2 = @(tau) abs(real(func1(tau)));
    else
        tau_res = 0.01/std(f_c);
        func2 = @(tau) abs(func1(tau));
    end
    
    [corr, C.tau(c)] = my_max_serach(func2, tau_hat_prime - opts.max_tau_err, ...
        tau_hat_prime + opts.max_tau_err, tau_res, opts.tau_fine_res, opts.grid_size);
    C.tauErr(c) = C.tau(c) - C.tau_exp(c);
    if opts.verbose
        fprintf("c = %3d\t\t", c);
        fprintf("Err before: %.2f usec\t\t", initial_error*1e6);
        fprintf("Err after: %.2f usec\n", C.tauErr(c)*1e6);
    end
    
    if opts.plot_flag
        figure("name", "c="+c);
        tau_grid = tau_hat_prime + linspace(-opts.max_tau_err, opts.max_tau_err, 100)';
        corr_grid = func2(tau_grid);
        plot(tau_grid*1e6, corr_grid);
        if ~C.isFA(c)
            xline(C.tau_exp(c)*1e6, "Color", "r");
        end
        hold on;
        plot(C.tau(c)*1e6, corr, "v");
        xline(tau_hat_prime*1e6, "Color", "g");
        mylegend("Corr", "exp", "est", "init");
        title(sprintf("c = %d, initial error = %.2f usec, final error = %.2f usec", c, initial_error*1e6, C.tauErr(c)*1e6));
        drawnow();
    end

end

%% update cluster statistics
[C, expected2] = phalcor.cluster_stats(C, ...
    opts.expected, opts.tauTol, opts.omegaTol); 

end

function [max_val, x_best] = my_max_serach(f, x1, x2, tol_init, tol_fin, grid_size)

tol = tol_init;
x = x1 : tol_init : x2;
while 1
    y = f(x);
    [max_val, i] = max(y);
    x_best = x(i);
    if tol <= tol_fin
        break
    end
    x = x_best + linspace(-tol, tol, grid_size);
    tol = x(2)-x(1);
end

end

