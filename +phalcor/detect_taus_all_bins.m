function T3 = detect_taus_all_bins(anm_stft, fVec, corners, Jt, Jf, tauMax, ...
    expected, detects_tau_name_val_pairs, opts)
% Author: Tom Shlomo, ACLab BGU, 2020

arguments
    anm_stft double
    fVec (:,1) double
    corners (:,2) double
    Jt (1,1) double
    Jf (1,1) double
    tauMax (1,1) double
    expected table
    detects_tau_name_val_pairs  cell
    
    opts.dirKnownFlag = false
    opts.nfftFactor = 1
    
    opts.plotFlag = 0
    opts.tVec
    opts.hopFactor
    opts.window
    opts.fs
    opts.smoothing_window = 'none'
    opts.use_trace = 0
end

N = sqrt(size(anm_stft, 3))-1;
nCorners = size(corners, 1);
df = fVec(2)-fVec(1);

omega_grid = sampling_schemes.fliege_maier(29);
Ygrid = shmat(N, omega_grid);
if opts.dirKnownFlag
    omegaDir = expected.omega(1,:);
    YgridDir = shmat(N, omegaDir);
else
    omegaDir = omega_grid;
    YgridDir = Ygrid;
end
H = wbar();
T3 = cell(nCorners);

for i=1:nCorners
    %% time smoothing + phase alignment
    fInd = corners(i,1)+(0:Jf-1);
    tInd = corners(i,2)+(0:Jt-1);
    anm_stft_slice = anm_stft(fInd, tInd, :);
    [Rbar, tauVec] = smoothing_stft(anm_stft_slice, 'TimeSmoothingWidth', inf, 'fVec', fVec(fInd), 'tauMax', tauMax , ...
        'permute', true, 'nifft', Jf*opts.nfftFactor, 'usetrace', opts.use_trace, 'window', opts.smoothing_window);
    
    %%
    Ttmp = phalcor.detect_taus(Rbar, tauVec, ...
        'expected', expected,...
        'plotFlag', opts.plotFlag==2, ...
        'omegaGridRef', omega_grid, ...
        'YGridRef', Ygrid, ...
        'omegaGridDir', omegaDir, ...
        'YGridDir', YgridDir,...
        detects_tau_name_val_pairs{:});
   
    if ~isempty(Ttmp)
        [~,Ttmp.idxInBin] = sort(Ttmp.rhoDir, 'descend');
        Ttmp{:,"i"} = i;
        Ttmp{:,"f"} = fVec(corners(i,1));
        if isfield(opts, "tVec")
            Ttmp{:,"t"} = opts.tVec(corners(i,2));
        end
        T3{i} = Ttmp;
    end
    
    % update waitbar every 10 iterations
    if mod(i,10)==1 || i==nCorners
        wbar(i, nCorners, H);
    end
end
T3 = vertcat(T3{:});

%% plot results
if opts.plotFlag
    figure("name", "Direct DOA vs. Delay");
    scatter3(T3.omegaDir(:,1)*180/pi, T3.omegaDir(:,2)*180/pi, T3.tau*1000, 200, T3.rhoDir, '.');
    hold on; plot3(expected.omega(:,1)*180/pi, expected.omega(:,2)*180/pi, expected.delay*1000, 'o', 'MarkerSize', 25);
    xlabel('$\theta_{dir}$ [deg]');
    ylabel('$\phi_{dir}$ [deg]');
    zlabel('$\tau$ [misec]');
    title('all');
end