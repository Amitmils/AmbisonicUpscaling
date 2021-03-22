function [R, tauVec] = smoothing_stft(P, varargin)
% Author: Tom Shlomo, ACLab BGU, 2020

%% defaults and input parsing
Q = size(P,3);
T = size(P,2);
F = size(P,1);
focusingMatrices = [];
timeSmoothingType = 'none';
freqSmoothingType = 'none';
nifft = F;
fVec = [];
tauMax = inf;
permuteFlag = false;
dtau = inf;
windowFlag = false;
window = [];
use_trace_flag = false;
for i=1:2:length(varargin)
    name = varargin{i};
    val = varargin{i+1};
    switch lower(name)
        case {'focusing','focusingmatrices','focusingmatrix','focusingmatrixes'}
            focusingMatrices = val;
        case {'timesmoothingwidth','timewidth', 'nt'}
            k_time = val;
            if ( isscalar(k_time) && k_time>1 ) || ( isvector(k_time) && any(k_time>0) )
                timeSmoothingType = 'movmean';
            end
        case {'freqsmoothingwidth','frequencysmoothingwidth','freqwidth','frequencywidth'}
            k_freq = val;
            if k_freq>1
                freqSmoothingType = 'movmean';
            end
        case {'alpha'}
            alpha = val;
            timeSmoothingType = 'iir';
        case {'ifft', 'nifft'}
            nifft = val;
            freqSmoothingType = 'ifft';
        case {'fvec','f'}
            fVec = val;
        case {'dtau'}
            dtau = val;
            freqSmoothingType = 'ifft';
        case {'taumax','maxtau'}
            tauMax = val;
        case {'permute', 'permuteflag'}
            permuteFlag = val;
        case {"usetrace"}
            use_trace_flag = val;
        case {'window'}
            windowFlag = ~strcmp(val, "none");
            window = val;
        otherwise
            error('unknown option: %s', name);
    end
end

%% apply focusing
if ~isempty(focusingMatrices)
    assert(size(focusingMatrices,3)==F, 'size(focusingMatrices,3) (number of focusing matrices) must be the same as size(P,1) (number of frequencies)');
    assert(size(focusingMatrices,2)==Q, 'size(focusingMatrices,2) must be the same as size(P,3) (number of channels)');
    Qold = Q;
    Q = size(focusingMatrices,1);
    % add layers to P (if needed)
    if size(focusingMatrices,1)>Q
        P = cat(3, P, zeros(F, T, Q-Qold));
    end
    % apply focusing matrices for each band
    for f=1:F
        P(f, :, 1:Q) = ipermute( focusingMatrices(:,:,f) * permute( P(f,:,:), [3 2 1] ), [3 2 1] );
    end
    % remove extra layers (if the number of channels was larger before
    % focusing)
    P(:,:,Q+1:end) = [];
end

%% generate flattened, rank-1 estimates
q = (1:Q)';
pairs = [q q; nchoosek(q, 2)];
R1 = P(:, :, pairs(:,1)) .* conj( P(:, :, pairs(:,2)) );
clear P;

%% Time smoothing
% tic;
switch timeSmoothingType
    case 'movmean'
        if any(k_time == inf)
            R1 = mean(R1, 2);
            T = 1;
        else
            if isscalar(k_time)
                k_time = [0 k_time-1];
            end
            R1 = movmean(R1, k_time, 2);
        end
    case 'iir'
        R1 = filter(alpha, [1, alpha-1], R1, [], 2);
    case 'none'
        
    otherwise
        error('unknown timeSmoothingType: %s', timeSmoothingType);
end
% fprintf('time smoothing time: %.3f sec\n', toc);

%% Frequency smoothing
switch freqSmoothingType
    case 'movmean'
        if use_trace_flag
            tr = sum(R1(:, :, pairs(:,1)==pairs(:,2)), 3);
            R1 = R1./tr;
        end
        if isequal(k_freq, inf)
            R1 = mean(R1, 1);
            F = 1;
        else
            R1 = movmean(R1, k_freq, 1);
        end
        %         fprintf('frequency smoothing time: %.3f sec\n', toc);
        %         flipFlag = false;
    case 'ifft'
        %         tic;
        %         R1 = ifft(R1, nifft, 1);
        %         fprintf('ifft time: %.3f sec\n', toc);
        %         F = nifft;
        %         flipIdx = [1, F:-1:2];
        %         flipFlag = true;
    case 'none'
        %         flipFlag = false;
    otherwise
        error('unknown freqSmoothingType: %s', freqSmoothingType);
end

%% unflatten to square matrices
R = zeros(F, T, Q, Q);
% tic;
% ind = sub2ind([Q Q], pairs(:,1), pairs(:,2));
% R(:,:,ind) = R1;
% toc;
% tic;
for i=1:size(pairs,1)
    R(:,:,pairs(i,1), pairs(i,2)) = R1(:,:,i);
    if i>Q
        %         if flipFlag
        %             R(:,:,pairs(i,2), pairs(i,1)) = conj(R1(flipIdx,:,i));
        %         else
        R(:,:,pairs(i,2), pairs(i,1)) = conj(R1(:,:,i));
        %         end
    end
end
% fprintf('unflatting time: %.3f sec\n', toc);

%%
switch freqSmoothingType
    case 'movmean'
        
    case 'ifft'
        %         tic;
        if ~isempty(fVec)
            df = fVec(2)-fVec(1);
            nifft = 2^nextpow2(max(nifft, 1/(dtau*df)));
        else
            nifft = 2^nextpow2(nifft);
        end
        %         if isempty(dtau)
        %
        %         elseif isempty(dtau) && ~isempty(nifft)
        %             assert(~isempty(fvec));
        %             df = fVec(2)-fVec(1);
        %             nifft = 2^nextpow2(max(F, 1/(dtau*df)));
        %         elseif isempty(nifft) && isempty(dtau)
        %             nifft = 2^nextpow2(F);
        %         end
        if use_trace_flag
            tr = sum(R(sub2ind(size(R), repmat((1:F)', 1, Q), ones(F,Q), repmat(1:Q, F, 1), repmat(1:Q, F, 1))), 2);
            R = R./tr;
        end
        if windowFlag
            if isfunc(window)
                window = window(F);
            end
            R = R.*window;
        end
        R = ifft(R, nifft, 1);
        %         fprintf('ifft time: %.3f sec\n', toc);
        if ~isempty(fVec)
            assert(F==length(fVec));
            df = fVec(2)-fVec(1);
            dtau = 1/(nifft*df);
            tauVec = (0 : nifft-1)'*dtau;
            I = find(tauVec <=tauMax, 1, 'last');
            tauVec = tauVec(1:I);
            R = R(1:I, :, :, :);
            R = R .* exp(1i*2*pi*fVec(1)*tauVec);
        elseif nargout>=2
            error("to output tauVec, you must provide fVec");
        end
        
    case 'none'
        
    otherwise
        error('unknown freqSmoothingType: %s', freqSmoothingType);
end
if permuteFlag
    R = permute(R, [3 4 1 2]);
end
end

