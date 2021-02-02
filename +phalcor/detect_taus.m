function  [T, fig] = detect_taus(R, tauVec, opts)
% Author: Tom Shlomo, ACLab BGU, 2020

arguments
    R double
    tauVec (:,1) double
    opts.peaksVec (1,1) string {mustBeMember(opts.peaksVec, ["none", "gammaDir", "gamma", "rhoDir", "rho"])} = "none"
    
    opts.expected table = table(nan(1), nan(1,2), 'VariableName', ["delay", "omega"]);
    opts.plotFlag (1,1) logical = false
    
    opts.minRhoDir (1,1) double {mustBeNonnegative, mustBeLessThanOrEqual(opts.minRhoDir, 1)} = 0
    opts.minRho (1,1) double {mustBeNonnegative, mustBeLessThanOrEqual(opts.minRho, 1)} = 0
    opts.peakEnv (1,1) double {mustBeNonnegative} = 0
    
    opts.newtonFlag (1,1) logical = false
    opts.newtonTol (1,1) double {mustBeNonnegative} = 0.01*pi/180;
    opts.newtonMaxIter (1,1) double {mustBePositive, mustBeInteger} = 50;
    
    opts.omegaGridRef (:,2) double = [];
    opts.YGridRef double = [];
    opts.omegaGridDir (:,2) double = [];
    opts.YGridDir double = [];
end

%% size, defaults and argument checking
Q = size(R,1);
N = ceil(sqrt(Q)-1);
ntau = length(tauVec);
dtau = tauVec(2)-tauVec(1);
if isempty(opts.omegaGridRef)
    opts.omegaGridRef = sampling_schemes.fliege_maier(29);
end
if isempty(opts.YGridRef)
    opts.YGridRef = shmat(N, opts.omegaGridRef);
end
if isempty(opts.omegaGridDir)
    opts.omegaGridDir = opts.omegaGridRef;
    opts.YGridDir     = opts.YGridDir;
elseif isempty(opts.YGridDir)
    opts.YGridDir = shmat(N, opts.omegaGridDir);
end
assert(size(opts.YGridDir,1)==size(opts.omegaGridDir,1));
assert(size(opts.YGridRef,1)==size(opts.omegaGridRef,1));
if isfield(opts, "expected")
    [~,dir_idx_exp] = min(opts.expected.delay);
    omegaDirExp = opts.expected.omega(dir_idx_exp, :);
else
    omegaDirExp = [nan nan];
end
if opts.plotFlag
    fig = figure("name", "rhos");
    tl = tiledlayout(3,1, "Padding", "none", "TileSpacing", "none");
else
   fig = [];
end

%% rho, gamma
[U,S,V] = svdnd(R, [1 2], [], 0);
%         fprintf("SVD time: %.3f sec\n", toc());
u = reshape(U(:,1,:), Q, ntau);
v = reshape(V(:,1,:), Q, ntau);
s = reshape(S(1,:,:), 1, []).';

[rhoDir, omegaDir] = sphere_max_abs(v, ...
    "newtonFlag", opts.newtonFlag, ...
    "omegaGrid", opts.omegaGridDir, ...
    "Ygrid", opts.YGridDir, ...
    "newtonMaxIter", opts.newtonMaxIter, ...
    "newtonTol", opts.newtonTol, ...
    "normalization", "rho");
rhoDir = rhoDir.';
omegaDirErr = angle_between(omegaDir, omegaDirExp);
gammaDir = s.*rhoDir;


if opts.peaksVec == "gamma" || opts.peaksVec == "rho" || opts.minRho>0 || opts.plotFlag
    [rhoRef, omegaRef] = sphere_max_abs(u, ...
        "newtonFlag", opts.newtonFlag, ...
        "omegaGrid", opts.omegaGridRef, ...
        "Ygrid", opts.YGridRef, ...
        "newtonMaxIter", opts.newtonMaxIter, ...
        "newtonTol", opts.newtonTol, ...
        "normalization", "rho");
    rhoRef = rhoRef.';
    rho = rhoDir.*rhoRef;
    gamma = s.*rho;
    extraVarsFlag = true;
else
    extraVarsFlag = false;
end


if opts.plotFlag
    ax(1) = nexttile(tl);
    h1 = plot(tauVec*1e3, [rho, rhoDir, rhoRef], '.-', "Parent", ax(1));
    xlines(opts.expected.delay*1e3);
    mylegend("$\rho$", "$\rho_{dir}$", "$\rho_{ref}$");
    ax(2) = nexttile(tl);
    h2 = plot(tauVec*1e3, [gamma gammaDir], '.-', "Parent", ax(2));
    xlines(opts.expected.delay*1e3);
    mylegend("$\gamma$", "$\gamma_{dir}$");
    ax(3) = nexttile(tl);
    plot(tauVec*1e3, omegaDirErr*180/pi, '.-', "Parent", ax(3));
    xlines(opts.expected.delay*1e3);
    ylabel('omega err');
    
    linkaxes(ax, 'x');
    xlim([0 tauVec(end)*1e3]);
    switch opts.peaksVec
        case "rhoDir"
            h = h1(2);
        case "gamma"
            h = h2(1);
        case "rho"
            h = h1(1);
        case "gammaDir"
            h = h2(2);
    end
    h.LineWidth = h.LineWidth*2;
end


%% find peaks
if opts.peaksVec=="none"
    locs = (1:ntau)';
    if opts.minRho == 0 && opts.minRhoDir == 0
        warning("peaksVec=none, but minRho and minRhoDir are both 0. This would result in many detections, and is probably not what you want.");
    end
else
    switch opts.peaksVec
        case "rhoDir"
            peaksVec = rhoDir;
        case "gamma"
            peaksVec = gamma;
        case "rho"
            peaksVec = rho;
        case "gammaDir"
            peaksVec = gammaDir;
        case "none"
            peaksVec = [];
    end
    if opts.peakEnv==0
        locs = (1:ntau)';
    else
        locs = find(peaksVec == movmax(peaksVec, max(floor(opts.peakEnv/dtau/2)*2+1, 3))); % finds local maximas
    end
end
%% create table
T = table();
T.tau = tauVec(locs);
T.rhoDir = rhoDir(locs);
T.gammaDir = gammaDir(locs);
T.omegaDir = omegaDir(locs, :);
T.tauIdx = locs;
T.u = u(:,locs).';
T.v = v(:,locs).';
T.sigma = s(locs);

if extraVarsFlag
    T.omegaRef = omegaRef(locs, :);
    T.gamma = gamma(locs);
    T.rhoRef = rhoRef(locs);
    T.rho = rho(locs);
end

% add some additional rows
T{:,"rhoDir0"} = rhoDir(1);
T = sortrows(T, opts.peaksVec, 'descend');

%% filter peaks
if opts.minRhoDir > 0
    T(T.rhoDir < opts.minRhoDir, : ) = [];
end
if opts.minRho > 0
    assert( ismember(opts.peaksVec, ["rho", "gamma"]),...
        "This test makes sense only if rho is calculated. rho is calculated only if peaksVec is rho, gamma");
    T(T.rho < opts.minRho, : ) = [];
end
%% plot peaks
if opts.plotFlag
    hold(ax(1), 'on');
    plot(T.tau*1e3, [T.rho T.rhoDir T.rhoRef], 'r.', 'MarkerSize', 16, 'Parent', ax(1));
    
    hold(ax(2), 'on');
    plot(T.tau*1e3, [T.gamma T.gammaDir], 'r.', 'MarkerSize', 16, 'Parent', ax(2));
end

end

