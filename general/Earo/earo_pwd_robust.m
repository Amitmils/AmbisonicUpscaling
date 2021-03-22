function [varargout] = earo_pwd_robust(eobj, N, param, method_flag,debug_flag)
% function [anm, [A],[f_vec]] = earo_pwd_robust(eobj, N, [param])
%
% Perform Plane-Wave Decomposition on an EARO object
%
%   eobj        - Input EARO object
%   [N]         - (optional) Decomposition order
%   [param]     - (optional) may contain te following fields:
%                   'a_max':radial filter soft-limiting factor in dB, {default = inf = no limiting}
%                   'th_grid,ph_grid':angles for space domain a(\theta,\phi) calculations
%                   'Ns': the assumed sound field order (higher than array order)
%                   'Raa': autocorrelation matrix of estimated anm (for future research)
%                   'Snn': {not in use}
%                   'sig_s': {not in use}
%   [method_flag] - (optional) determines the PWD method
%                   '1': standard PWD (can be reularized using bernshultz)
%                   '2': BroadBand PWD - incorporates both R-PWD ('3') and PWD-AC ('5')
%                           C=inv(D*D^H+(1/alpha)*inv(diag(b)*diag(b)'))
%                   '3': Regularized PWD with regularization method according to "param.regularization"
%                       For example:
%                           0:    No regularization - maxDI or modal beamforming or conventional PWD
%                           1:    RPWD regularization 
%                               with optimal regularized radial function
%                               C=inv((1/alpha)*inv(diag(b)*diag(b)'))
%                           11:   RPWD regularization + normalization
%                           3:    max WNG beamforming + normalization
%                           31:   max WNG beamforming
%                           4:    Tikhonov Regularization
%                           5:    Soft-limiting                   
%                   '4': aliasing cancellation up to, Ns, the soundfield order
%                           C=D^H*inv(D*D^H)
%                   '5': aliasing cancellation up to, Na, the Array order
%                           C=inv(D*D^H)
%                   '6': unused here (it is used outside the function in original code to get normalized output...)
%                   '7': aliasing cancellation up to, Na, the Array order
%                       with respect to, Raa, the autocorellation of the
%                       soundfield:  C=inv(D*D^H)
%                   '8':same as '4' with eualiztion factor
%                           C=(Ns+1)^2/trace(D^H*inv(D*D^H)*D)D^H*inv(D*D^H)
%
%   Output:
%   anm         - length(f) x (N+1)^2 matrix of plane wave density in the
%                   SH domain.
%   [A]         - (optional) length(f) x length(az/el) matrix of plane wave
%                   density coefficients in the space domain; computed over
%                   the azimuth/elevation grid specified in eobj.micGrid
%   [f]         - 1 x length(f) vector containing the frequencies [Hz] in which
%                   the other outputs are calculated
%
% This function operates over the eobj.micGrid dimension.
%
% August 2014, Jonathan Sheaffer and David Alon, Ben-Gurion University
% Part of the EARO Beamforming toolbox
%

useSOFiA = false;                     % change to true if you wish to generate the radial functions using SOFiA toolbox
% instead of EARO's internal func.


if (nargin<2) && (~isempty(eobj.orderN)), N=eobj.orderN; end;          % Default = Decomposition order is the same as microphone array order
if nargin<4,
    if nargin<3,
        method_flag=1;
        param.a_max=inf;          % Default = no soft-limiting or any other robustnss
    else
        method_flag=2;      %use soft-limiting
        if isstruct(param)
            a_max=param.a_max;        % soft-limiting max amplification
        else
            a_max=param(1);        % soft-limiting max amplification
        end
    end
end;
if nargin<5, debug_flag=0; end


% Transform to frequency domain, if needed
if strcmp(eobj.dataDomain{1},'TIME')
    [eobj,fVec]=eobj.toFreq();
else
    nFFT=size(eobj.data,2);
    fVec = (0:(nFFT-1))*(eobj.fs/nFFT);
end

% Compute speed of sound
if ~isempty(eobj.avgAirTemp) && isnumeric(eobj.avgAirTemp)
    c = 331.3 * sqrt(1+(eobj.avgAirTemp/273.15));
else
    c = 343.5; % default value;
    
end

% Construct kr vector
fVec=double(fVec);
kr=double(fVec*2*pi*eobj.micGrid.r(1)/c);
f_aliasing=c*eobj.orderN/(2*pi*eobj.micGrid.r(1));


% Generate radial functions
if eobj.scatterer, sphereType=1; else sphereType=0; end;
if exist('param','var')
    if  isstruct(param),
        a_max=param.a_max;
    else
        a_max=param(1);
    end
else
    a_max=inf;
end;

if method_flag==3 || method_flag==10
    %for method_flag==3 the param.a_max field refer to the mode
    %amplification stage and not the radial function stage
    bn = radialMatrix(N,kr,sphereType,Inf,true,useSOFiA);
else
    bn = radialMatrix(N,kr,sphereType,a_max,true,useSOFiA);
end
if method_flag~=10,
    % Transform to SH domain, if needed; if method==10 (Craig jin method) than stay in space domain
    if strcmp(eobj.dataDomain{2},'SPACE')
        eobj=eobj.toSH(N,'MIC');
    elseif strcmp(eobj.dataDomain{2},'SH')
        if ~(eobj.orderN==N)  % if N and orderN are not the same, then fix it
            eobj=eobj.toSpace('MIC');
            eobj=eobj.toSH(N,'MIC');
        end
    end
end

%% Calculating matrix C to transform from pressure coefficients to PW coefficients
% At this point the pressure should be represented in SH domain and therefore
%a_nm(k)=C*diag(b)^(-1)*p_nm(k)=R_f*p_nm(k)
switch method_flag,
    case 10
        Yq=(sh2(eobj.orderN, eobj.micGrid.elevation, eobj.micGrid.azimuth)).';
        Yqs=(sh2(param.Ns, eobj.micGrid.elevation, eobj.micGrid.azimuth)).';
        bns = radialMatrix(param.Ns,kr,sphereType,Inf,true,useSOFiA);
        bna = radialMatrix(N,kr,sphereType,Inf,true,useSOFiA);
        for f_ind=1:length(kr)
            if kr(f_ind)>0,
                %                   D=[eye(25),zeros(25,(81-25))];
                beta=1; % calculated assuming alpha=PWDNA=100=20dB => beta^2=1/(4*alpha)
                Tl=Yqs*diag(bns(f_ind,:));
                TL=Yq*diag(bna(f_ind,:));
                R_f=TL'*inv(Tl*Tl'+beta^2*eye(size(eobj.micGrid.elevation,2)));
            else
                R_f=zeros(size(Yq,2),size(Yq,1));
                R_f_tmp=Yq'*inv(Yqs*Yqs');  R_f(1,1)=R_f_tmp(1,1)/bna(f_ind,1);
            end
            anm(f_ind,:)=(R_f*double(squeeze(eobj.data(1,f_ind,:)))).';
        end
        if nargout>3  %if rfequested by user, output the mode amplification matrix
            varargout{4}=bn*0;
        end
        if nargout>4  %if rfequested by user, output the mode amplification matrix
            varargout{5}=bn;
        end
    case 3
        % Perform PWD with desired regulariztion (default is MMSE Robust-PWD regulariztion)
        %       debug_flag=1;
        if ~isfield(param,'regularization')
            param.regularization=1; %default R-PWD regularization
            Q=2*(N+1)^2; param.alpha=10*Q/(4*pi)^2; %assumes samplimng scheme satisfes Q=2*(N+1)^2 and SNR=+10dB at all freqs
        end
        C_f=modeAmplificationMatrix(bn,kr,param,param.regularization);
        anm=(squeeze(sum(C_f.*repmat(permute(eobj.data(1,:,:),[1,3,2]),[(N+1)^2,1,1]),1))).';
        if nargout>3  %if rfequested by user, output the mode amplification matrix
            varargout{4}=C_f;
        end
        if nargout>4  %if rfequested by user, output the mode amplification matrix
            varargout{5}=bn;
        end
        
    case {2,4,5,7,8,9}
        switch method_flag
            case {2}  % BB-PWD: both PWD-AC and R-PWD
                if ~isfield(param,'alpha')
                    Q=2*(N+1)^2; param.alpha=10*Q/(4*pi)^2; %assumes samplimng scheme satisfes Q=2*(N+1)^2 and SNR=+10dB at all freqs
                end
                warning('off','MATLAB:nearlySingularMatrix');
                display('MATLAB warning :nearlySingularMatrix was turned off');
                % desired result is the array order PWD- but R_f is higher dimension
                D_Na=eye((eobj.orderN+1)^2);
                %D_Na=[eye((eobj.orderN+1)^2),zeros((eobj.orderN+1)^2,(param.Ns+1)^2-(eobj.orderN+1)^2)];
                anm=zeros(length(kr),(N+1)^2);
            case {5,9} % desired result is the array order PWD
                D_Na=[eye((eobj.orderN+1)^2),zeros((eobj.orderN+1)^2,(param.Ns+1)^2-(eobj.orderN+1)^2)];
                anm=zeros(length(kr),(param.Ns+1)^2);
            otherwise
                anm=zeros(length(kr),(param.Ns+1)^2);
        end
        Yq=(sh2(eobj.orderN, eobj.micGrid.elevation, eobj.micGrid.azimuth)).';
        Yqs=(sh2(param.Ns, eobj.micGrid.elevation, eobj.micGrid.azimuth)).';
        bns = radialMatrix(param.Ns,kr,sphereType,Inf,true,useSOFiA);
        bna = radialMatrix(N,kr,sphereType,Inf,true,useSOFiA);
        bnainv=bna.^(-1);
        if ~isfield(param,'Raa'), param.Raa=repmat(eye((param.Ns+1)^2),[1,1,length(kr)]); end
        %         Raa_eq=param.Raa;  %seperate processing for the spectral equalization and AC-spatial solution
        %         Raa=eye((param.Ns+1)^2);
        pinvYq=pinv(Yq);
        Raa=param.Raa(:,:,1);
        C_f=zeros((N+1)^2,(N+1)^2,length(kr)); D_f=zeros((N+1)^2,(param.Ns+1)^2,length(kr));
        fprintf('decomposing into plane waves: f_ind=        ');
        for f_ind=1:length(kr)
            if size(param.Raa,3)==length(kr), Raa=param.Raa(:,:,f_ind); end
            if kr(f_ind)>0,
                %                   D=[eye(25),zeros(25,(81-25))];
                D=diag(bnainv(f_ind,:))*pinvYq*Yqs*diag(bns(f_ind,:));
                switch method_flag
                    case {2}  % BB-PWD: both PWD-AC and R-PWD
                        C=D_Na'*inv(D*D'+(1/param.alpha)*(diag(bnainv(f_ind,:).*conj(bnainv(f_ind,:)))));
                        R_f=C*diag(bnainv(f_ind,:));
                        if nargout>3  %if rfequested by user, output the mode amplification matrix
                            C_f(:,:,f_ind)=C;%cat(3,C_f,);
                            D_f(:,:,f_ind)=D;%cat(3,D_f,);
                        end
                    case {5,9} % desired result is the array order PWD
                        %                   R_f=D_Na'*inv(D*D')*diag(bn(f_ind,:).^-1);
                        
                        Raa_Na=zeros((param.Ns+1)^2); Raa_Na(1:((eobj.orderN+1)^2),:)=Raa(1:((eobj.orderN+1)^2),:);
                        R_f=Raa_Na*D'*inv(D*Raa*D')*diag(bn(f_ind,:).^-1);
                    otherwise % desired result is the sound field order PWD (Ns)
                        R_f=Raa*D'*inv(D*Raa*D')*diag(bn(f_ind,:).^-1);
                end
            else
                switch method_flag
                    case {2}  % BB-PWD: both PWD-AC and R-PWD
                        R_f=zeros((N+1)^2);  R_f(1)=bnainv(f_ind,1);
                            C_f(1,1,f_ind)=1;%cat(3,C_f,);
                            D_f(1:(N+1)^2,1:(N+1)^2,f_ind)=eye((N+1)^2);%cat(3,D_f,);
                    otherwise
                        R_f=zeros((param.Ns+1)^2,(N+1)^2);  R_f(1)=bn(f_ind,1).^-1;
                end
            end
            anm(f_ind,:)=(R_f*double(squeeze(eobj.data(1,f_ind,:)))).';
            if method_flag==8 && kr(f_ind)>0,
                %                 Raa_eq=anm(f_ind,:).'*conj(anm(f_ind,:));
                %                 a_equalization=sqrt((trace(Raa_eq))/trace(Raa_eq*D'*inv(D*Raa*D')*D*Raa'));
                %                 a_equalization_vec(f_ind)=sqrt(trace(D'*(diag(abs(bna(f_ind,:)).^2))*D)/trace(D'*inv(D*D')*D));
                a_equalization_vec(f_ind)=(trace(D'*(diag(abs(bna(f_ind,:)).^2))*D)/trace(D'*inv(D*D')*D));
                %!!!!! only part of Raa are considered here for the
                %equalization!!!! note different "Raa" and "Raa_eq"
                %                       R_f=((param.Ns+1)^2)/trace(D'*inv(D*D')*D)*R_f;
                %                 anm(f_ind,:)=a_equalization*anm(f_ind,:);
            end
            if method_flag==9 && kr(f_ind)>0,
                %                 Raa_eq=anm(f_ind,:).'*conj(anm(f_ind,:));
                %                 a_equalization=sqrt((trace(Raa_eq))/trace(Raa_eq*D'*inv(D*Raa*D')*D*Raa'));
                a_equalization_vec(f_ind)=(N+1)/sqrt(trace(inv(D*D')));
                %!!!!! only part of Raa are considered here for the
                %equalization!!!! note different "Raa" and "Raa_eq"
                %                       R_f=((param.Ns+1)^2)/trace(D'*inv(D*D')*D)*R_f;
                anm(f_ind,:)=a_equalization_vec(f_ind)*anm(f_ind,:);
            end
            
            if mod(f_ind,100)==0,
                for print_ind=[0:floor(log10(f_ind))], fprintf('\b');  end;     fprintf('%d', f_ind); % delete previous counter display and than display new
            end
            %[P_stft{mic_index} num_freq num_frames] = STFT(p_t(mic_index,:).',window,overlap,frame_length,M);
        end
        fprintf('\n');

                if method_flag==2,
                    anm=[anm,zeros(length(kr),(param.Ns+1)^2-(N+1)^2)];
                warning('on','MATLAB:nearlySingularMatrix');
                display('MATLAB warning :nearlySingularMatrix was turned back on');
                        if nargout>3  %if rfequested by user, output the mode amplification matrix
                            varargout{4}=C_f;
                            if nargout>4  %if rfequested by user, output the mode amplification matrix
                                varargout{5}=bn;
                            end
                            if nargout>5  %if rfequested by user, output the mode amplification matrix
                                varargout{6}=D_f;
                            end
                        end
                end
        if method_flag==8,
            kr_mean_ind=[2:find(kr<=N/10,1,'last')];
            C_equaliztion=(1/mean(a_equalization_vec(kr_mean_ind)));%*(param.Ns+1)/(N+1);
            anm=C_equaliztion*repmat(a_equalization_vec.',1,(param.Ns+1)^2).*anm;
        end
        %           kr(f_ind)>0
        
    case {1}
        % Perform PWD with desired SL regulariztion
        %         for f_ind=1:length(kr)
        %             if kr(f_ind)>0,
        %                 %                   D=[eye(25),zeros(25,(81-25))];
        %                 R_f=diag(bn(f_ind,:).^-1);
        %             else
        %                 R_f=zeros((N+1)^2,(N+1)^2);%  R_f(1)=bn(f_ind,1).^-1;
        %             end
        %             anm(f_ind,:)=(R_f*double(squeeze(eobj.data(1,f_ind,:)))).';
        %         end
        if kr(1)==0,
            R_m=[zeros(1,(N+1)^2);bn(2:end,:).^-1];
        else
            R_m=bn.^-1;
        end
        anm=(R_m.*double(squeeze(eobj.data(1,:,:))));
        
    case 0
        % Perform PWD with maxWNG regulariztion
        bna = radialMatrix(N,kr,sphereType,Inf,true,useSOFiA);
        for f_ind=1:length(kr)
            if kr(f_ind)>0,
                %                   D=[eye(25),zeros(25,(81-25))];
                R_f=diag((abs(bna(f_ind,:)).^2).*bna(f_ind,:).^-1);
            else
                R_f=zeros((N+1)^2,(N+1)^2);  R_f(1)=(abs(bna(f_ind,1)).^2).*bna(f_ind,1).^-1;
            end
            anm(f_ind,:)=(R_f*double(squeeze(eobj.data(1,f_ind,:)))).';
        end
        
    otherwise
        % Perform PWD (maxDI) without regulariztion
        bna = radialMatrix(N,kr,sphereType,Inf,true,useSOFiA);
        for f_ind=1:length(kr)
            if kr(f_ind)>0
                R_f=diag(bna(f_ind,:).^-1);
            else
                R_f=zeros((N+1)^2);  R_f(1)=bna(f_ind+1,1).^-1;
            end
            anm(f_ind,:)=(R_f*double(squeeze(eobj.data(1,f_ind,:)))).';
        end
end

if debug_flag && (method_flag==8 || method_flag==9),
    figure; plot(fVec,a_equalization_vec); title('Equaliztion factor Vs. frequency'); xlabel('f [Hz]');
    xlim([0,10000]);
end

% Output
varargout{1} = anm;

if nargout>1          % transform to space domain as well
    if method_flag==2 || method_flag==4 || method_flag==5|| method_flag==7|| method_flag==8|| method_flag==9
        Ntmp=param.Ns;
    else
        Ntmp=eobj.orderN;
    end
    if isstruct(param) && isfield(param,'th_grid')
        Y = shMatrix(Ntmp,param.th_grid,param.ph_grid);
    else
        Y = shMatrix(Ntmp,eobj.micGrid.elevation,eobj.micGrid.azimuth);
    end
    varargout{2} = anm*Y;
end

if nargout>2  %if rfequested by user, adding the frequency vector
    varargout{3}=fVec;
end

end

