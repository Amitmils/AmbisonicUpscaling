function [C, T3, expected2, intermediate_variables, hyperparams] = wrapper(anm_stft, f_vec, t_vec, stft_window_length, opts)

% Author: Tom Shlomo, ACLab BGU, 2020
arguments
    anm_stft (:,:,:) double % frequency x time x SH channel
    f_vec (:,1) double % Hz
    t_vec (:,1) double % sec
    stft_window_length % sec
    opts.expected table = table(zeros(0,1), zeros(0,2), "VariableNames", ["delay", "omega"]);
    
    opts.plotFlag (1,1) double = 0; % 0 (no plots), 1 (some plots), or 2 (all plots)
    
    % smoothing and phase alignment
    opts.bw_hz = 2000;
    opts.time_smoothing_sec = 400e-3;
    opts.fmin = 500;
    opts.fmax = 5000;
    opts.f_dilution = 8;
    opts.freq_smoothing_window = @(n) kaiser(n, 3)
    opts.nfft_factor = 2;
    opts.inverse_trace_weighting_flag = 1
    
    % bin-wise DOA and delay
    opts.peaksVec = "rhoDir"
    opts.taumax = 25e-3;
    opts.peakEnv = 0;
    opts.rhoThresh = 0.9;
    opts.omp_k_max = 3;
    opts.omp_tol = sqrt(0.4);
    
    % clustering
    opts.omegaDirHistTol = 10*pi/180;
    opts.tauWeight = 0.2e-3;
    opts.omegaWeight = 5*pi/180;
    opts.epsilon = 1;
    opts.densityThresh = 0.1;
    opts.taumin = 0
    
    % delay fine tuning
    opts.fine_delay_flag (1,1) logical = false
    
    % stats
    opts.omegaTol = 15*pi/180;
    opts.tauTol = 0.5e-3;
    
    %
    opts.intermediate_variables (1,1) struct = struct()
end

opts.expected.delay = opts.expected.delay - opts.expected.delay(1);
expected = opts.expected( opts.expected.delay < opts.taumax+opts.tauTol , : );
TIC0 = tic();
%% select bins
% freqs
fres = f_vec(2)-f_vec(1);
Jf = round(opts.bw_hz/fres);
freqs = find(f_vec>=opts.fmin, 1, 'first') : round(Jf/opts.f_dilution) : find(f_vec<=opts.fmax-opts.bw_hz, 1, 'last');

% times
hop = t_vec(2)-t_vec(1);
Jt = max(round((opts.time_smoothing_sec - stft_window_length)/hop + 1), 1);
times = 1:size(anm_stft,2)-Jt+1;

% construct corners array
[freqs, times] = ndgrid(freqs, times);
corners = [freqs(:), times(:)];
fprintf("num of corners: %d\n", size(corners,1));
fprintf("Jt = %d\n", Jt);
fprintf("Jf = %d\n", Jf);

%% phase alignment + delay detection
if ~isfield(opts.intermediate_variables, "T") && ~isfield(opts.intermediate_variables, "T2")
    TIC = tic();
    T = phalcor.detect_taus_all_bins(anm_stft, f_vec, corners, Jt, Jf, opts.taumax, ...
        expected, ...
        {"peaksVec", opts.peaksVec, ...
        "minRhoDir", opts.rhoThresh, ...
        "peakEnv", opts.peakEnv}, ...
        "plotFlag", opts.plotFlag, ...
        "use_trace", opts.inverse_trace_weighting_flag, ...
        "smoothing_window", opts.freq_smoothing_window, ...
        "nfftFactor", opts.nfft_factor);
    intermediate_variables.T = T;
    fprintf("Delay detection time: %.1f sec\n", toc(TIC));
elseif isfield(opts.intermediate_variables, "T")
    T = opts.intermediate_variables.T;
end

%% outlier removal + omp
if ~isfield(opts.intermediate_variables, "T2")
    TIC = tic();
    [T2, omegaDir] = phalcor.direct_sound_based_outlier_removal(T, "expected", expected, ...
        "omegaDirHistTol", opts.omegaDirHistTol, ...
        "rhoThresh", opts.rhoThresh, ...
        "plotFlag", opts.plotFlag);
    fprintf("direct DOA time: %.1f sec\n", toc(TIC));
    
    TIC = tic();
    T2 = phalcor.omp(T2, "expected", expected, ...
        "ompKmax", opts.omp_k_max, ...
        "ompTol", opts.omp_tol, ...
        "plotFlag", opts.plotFlag);
    fprintf("OMP time: %.1f sec\n", toc(TIC));
else
    T2 = opts.intermediate_variables.T2;
    omegaDir = opts.intermediate_variables.omegaDir;
end

%% clustering
if ~isfield(opts.intermediate_variables, "T3") || ...
        ~isfield(opts.intermediate_variables, "C_before_delay_fine_tune") || ...
        ~isfield(opts.intermediate_variables, "expected_before_delay_fine_tune")
    [C, T3, expected2] = phalcor.clustering(T2, omegaDir, "plotFlag", opts.plotFlag, ...
        "taumin", opts.taumin, ...
        "taumax", opts.taumax, ...
        "omegaWeight", opts.omegaWeight, ...
        "tauWeight", opts.tauWeight, ...
        "epsilon", opts.epsilon, ...
        "densityThresh", opts.densityThresh, ...
        "expected", expected, ...
        "tauTol", opts.tauTol, ...
        "omegaTol", opts.omegaTol);
else
    C = opts.intermediate_variables.C_before_delay_fine_tune;
    T3 = opts.intermediate_variables.T3;
    expected2 = opts.intermediate_variables.expected_before_delay_fine_tune;
end

%% cache intermediate variables
intermediate_variables.T2 = T2;
intermediate_variables.T3 = T3;
intermediate_variables.C_before_delay_fine_tune = C;
intermediate_variables.expected_before_delay_fine_tune = expected2;
intermediate_variables.omegaDir = omegaDir;
%% delay fine tuning
if opts.fine_delay_flag
    [C, expected2] = phalcor.fine_tune_delay(C, T3, Jf, fres, ...
        "expected", expected2, ...
        "plot_flag", opts.plotFlag == 2, ...
        "max_tau_err", 0.5/(Jf*fres), ...
        "real_flag", 1, ...
        "verbose", 1, ...
        "tau_fine_res", 0.001/opts.fmax, ...
        "tauTol", opts.tauTol, ...
        "omegaTol", opts.omegaTol);
end
fprintf("Total time: %.1f\n", toc(TIC0));
%% hyperparams
hyperparams = opts;
hyperparams = rmfield(hyperparams, ["plotFlag", "intermediate_variables", "expected"]);
hyperparams.Jf = Jf;
hyperparams.Jt = Jt;
hyperparams.frequency_resolution = fres;
hyperparams.stft_hop_sec = hop;
hyperparams.stft_window_length_sec = stft_window_length;
hyperparams.N = sqrt(size(T3.v, 2)) - 1;
hyperparams.anm_stft_size = size(anm_stft);
hyperparams.anm_stft_total_time = t_vec(2) - t_vec(1);

end

