
function [pStruct] = earo_simDRR_varN(hobj, Ns, roomDims, recPos, srcAngPos, srcType, freqAbs, alphaAbs, fs, isDiff, eqOrder)
% variable N, simulate direct to reverberant ratio

    rHead = 0.0875;
    c=343.5;

    if length(Ns)<2, Ns(2)=Ns(1); end;
    
    N=Ns(1); NE=Ns(2);     %N direct, NE early
    
    % collect variables
    srcAz = srcAngPos(3);
    srcEl = srcAngPos(2);
    srcDist = srcAngPos(1);

    % Set-up model
    absMatrix = repmat(alphaAbs, 6, 1);
    if isempty(freqAbs) || isempty(absMatrix)
        Room=SetupRoom('Dim',roomDims);
    else
        Room=SetupRoom('Dim',roomDims,'Freq',freqAbs,'Absorption',absMatrix);
    end
    Receivers=AddReceiver('Location',recPos,'Type','sphharm','MaxOrder',max(N,NE),'NFComp',true);
    [SrcX SrcY SrcZ] = sph2cart(srcAz,pi/2-srcEl,srcDist);
    Sources=AddSource([],'Location',[SrcX, SrcY, SrcZ] + recPos,'Type',srcType,'Orientation',[radtodeg(pi+srcAz),0,0]);

    
    if ~hobj.shutUp, PlotSimSetup(Sources,Receivers,Room); end;

    % Execute and convert SH types    
    simord = [8 8 8];
    
    Options=MCRoomSimOptions('SimDirect',true,'SimSpec',false,'SimDiff',false,'Fs',double(fs),'Order',simord);
    irD=RunMCRoomSim(Sources,Receivers,Room,Options);
    P=MCRoomPerm(max(N,NE))';
    C=SHc2r(max(N,NE))';
    anm_Direct=C*P*irD.';    

    Options=MCRoomSimOptions('SimDirect',true,'SimSpec',true,'SimDiff',isDiff,'Fs',double(fs),'Order',simord);
    irD=RunMCRoomSim(Sources,Receivers,Room,Options);
    P=MCRoomPerm(max(N,NE))';
    C=SHc2r(max(N,NE))';
    anm_Both=C*P*irD.';   

    anm_Direct(:,end:size(anm_Both,2))=0;
    anm_Earl = anm_Both - anm_Direct;
    
    % Do spatial lowpass on lower order component
    if NE<N
      anm_Earl(((NE+1)^2)+1:end,:)=0;
    elseif N<NE
        anm_Direct(((N+1)^2)+1:end,:)=0;
    end
    
    % FFT stuff
    nFFT = 2^nextpow2(size(anm_Both,2));
    if mod(nFFT,2), nFFT=nFFT+1; end;
    anm_Earl = fft(anm_Earl, nFFT, 2);
    anm_Direct = fft(anm_Direct, nFFT, 2);

    fVec = linspace(0,double(fs),nFFT);
    if strcmp(hobj.dataDomain{1},'FREQ'), hobj=hobj.toTime(); end;
    hobj = hobj.toFreq(nFFT);   

    % Trim negative frequencies 
    hobj.data(:,(nFFT/2)+2:end,:) = [];
    anm_Direct(:,(nFFT/2)+2:end,:) = [];
    anm_Earl(:,(nFFT/2)+2:end,:) = [];
    fVec((nFFT/2)+2:end) = [];
    
    % EQ and Merge
    if ~(eqOrder==-1)
        disp('Design timbre correction filter...');
        kr=2*pi*fVec*rHead/c;
        
        if NE<N
            sResp = meanRespN(NE, kr, true, true);
            tResp = meanRespN(eqOrder, kr, true, true);
            anm_Earl = anm_Earl.*repmat(tResp./sResp,1,size(anm_Earl,1)).';
        else
            sResp = meanRespN(N, kr, true, true);
            tResp = meanRespN(eqOrder, kr, true, true);
            anm_Direct = anm_Direct.*repmat(tResp./sResp,1,size(anm_Direct,1)).';  
        end
% %         
%         if NE<N
%             sResp = meanRespN(NE, kr, true, true);
%             tResp = meanRespN(eqOrder, kr, true, true);
%             anm_Earl = anm_Earl.*repmat(tResp./sResp,1,size(anm_Earl,1)).';
%         elseif NE>N
%             sResp = meanRespN(N, kr, true, true);
%             tResp = meanRespN(eqOrder, kr, true, true);
%             anm_Direct = anm_Direct.*repmat(tResp./sResp,1,size(anm_Direct,1)).';  
%         else
%             sResp = meanRespN(N, kr, true, true);
%             tResp = meanRespN(eqOrder, kr, true, true);
%             anm_Direct = anm_Direct.*repmat(tResp./sResp,1,size(anm_Direct,1)).';
%             anm_Earl = anm_Earl.*repmat(tResp./sResp,1,size(anm_Earl,1)).';
%         end
       
    end
    %anm=anm_Earl+anm_Direct;

    % SH transform
    hobj = hobj.toSH(max(N,NE),'SRC');
    
    % Iterate through head rotation angles
    if ~hobj.shutUp, fprintf('Generating BRIRs...\n'); end;
    tMat=tildize(max(N,NE));
    anm_tilde_Direct = anm_Direct.'*tMat;
    anm_tilde_Earl = anm_Earl.'*tMat;
    brirAngles=(pi/180)*(0:1:359);    
       
        Hnm_lt_rot=hobj.data(:,:,1);   %(hobj.data(:,:,1).'*D).';
        Hnm_rt_rot=hobj.data(:,:,2);   %(hobj.data(:,:,2).'*D).';   
        
        % Generate BRIR Direct
        for ii=1:size(anm_tilde_Direct,1)
            plk_D(ii)=anm_tilde_Direct(ii,:)*Hnm_lt_rot(:,ii);
            prk_D(ii)=anm_tilde_Direct(ii,:)*Hnm_rt_rot(:,ii);
        end
        plk_D(1)=real(plk_D(1));
        prk_D(1)=real(prk_D(1));
        plk_D(end)=real(plk_D(end));
        prk_D(end)=real(prk_D(end));
        plk_D=[plk_D,fliplr(conj(plk_D(2:end-1)))];
        prk_D=[prk_D,fliplr(conj(prk_D(2:end-1)))]; 
        
        % Generate BRIR Refl
        for ii=1:size(anm_tilde_Earl,1)
            plk_R(ii)=anm_tilde_Earl(ii,:)*Hnm_lt_rot(:,ii);
            prk_R(ii)=anm_tilde_Earl(ii,:)*Hnm_rt_rot(:,ii);
        end
        plk_R(1)=real(plk_R(1));
        prk_R(1)=real(prk_R(1));
        plk_R(end)=real(plk_R(end));
        prk_R(end)=real(prk_R(end));
        plk_R=[plk_D,fliplr(conj(plk_R(2:end-1)))];
        prk_R=[prk_D,fliplr(conj(prk_R(2:end-1)))]; 
    
    
    pStruct.ltD = plk_D;
    pStruct.rtD = prk_D;
    pStruct.ltR = plk_R;
    pStruct.rtR = prk_R;
    
end

%% Internal functions
function Perm=SHc2r(Nmax)

    % this code forms a permute matrix from the Normalized Complex Spherical Harmonics to
    % the Normalized Real Spherical Harmonics
    % Perm matrix hold the relation- Ynm_{Real} = Perm x Ynm_{Complex}

    Perm = zeros((Nmax+1)^2);
    sizeP = size(Perm,1);
    ind = 0;
    for n= 0:Nmax

        Perm((ind+1):(ind+1+(2*n+1)-1),(ind+1):(ind+1+(2*n+1)-1)) = miniSHc2r(n);
        ind = ind + (2*n +1);
    end

    Perm=conj(Perm);
   
end

function perm=miniMCRoomPerm(n)

    % a help function for MCRooPerm, permuting for each given n.

    perm = zeros((2*n+1));
    sizeP = size(perm,1);
    perm((floor(sizeP/2)+1),(2*n+1)) = 1;
    for ii= 1:(floor(sizeP/2))
        perm((floor(sizeP/2)+1-ii),(2*n+1) - 2*ii +1 ) = 1;
        perm((floor(sizeP/2)+1+ii),(2*n+1) - 2*ii ) = 1;
    end
end

function Perm=MCRoomPerm(Nmax)

    % this code forms a permute matrix that orders the coefficients that we use
    % to the order MCRoomSim does, following GenSHIndices.m The following does
    % so by C_{MCRoomSIM convention} = Perm x C_{our convention}

    Perm = zeros((Nmax+1)^2);
    sizeP = size(Perm,1);
    ind = 0;
    for n= 0:Nmax

        Perm((ind+1):(ind+1  +(2*n+1) - 1     ),(ind+1):(ind+1  +(2*n+1) - 1     )) = miniMCRoomPerm(n);
        ind = ind + (2*n +1);
    end

    Perm = inv(Perm); 
end

function perm=miniSHc2r(n)

    % a help function for SHc2r, permuting for each given n.

    perm = zeros((2*n+1));
    sizeP = size(perm,1);
    perm((floor(sizeP/2)+1),(floor(sizeP/2)+1)) = 1;
    for ii= 1:(floor(sizeP/2))
        perm((floor(sizeP/2)+1+ii),(floor(sizeP/2)+1+ii)) = 1/sqrt(2)*(-1)^ii;%*(-1)^ii;
        perm((floor(sizeP/2)+1+ii),(floor(sizeP/2)+1-ii)) = 1/sqrt(2);
        perm((floor(sizeP/2)+1-ii),(floor(sizeP/2)+1-ii)) = -1/(sqrt(2)*1j);%*(-1)^ii;
        perm((floor(sizeP/2)+1-ii),(floor(sizeP/2)+1+ii)) = +1/(sqrt(2)*1j)*(-1)^ii;
    end
end


function DM=WignerDM(N,alpha,beta,gamma)
% build Wigner D matrix (eq. (20), Ben Hagai Nov 2012) 
% first, gamma about z, then beta about y, then gamma about z again (counterclocwise)
DM=zeros((N+1)^2);
    for n=0:N
        for m=-n:n
            DM(n^2+n+m+1,n^2+1:(n+1)^2)=...
                wignerd(n,m,alpha,beta,gamma);
        end
    end
end

function D=wignerd(n,m,alpha,beta,gamma)
% wignerd.m
% ------------
%       Get Wigner-D Coefficients.
%       Based on Rafaely 2008, equations (11-12).
%
% Syntax
% ------------
%     D=wignerd(n,m,m2,alpha,beta,gamma)
%
% Input
% ------------
%     Required
%           n,m - scalar - the spherical indices
%           alpha,beta,gamma - Euler angles
%
% Output
% ------------
%         D - defined as D_{mm'}^n, where m'=-n:n
%
% Created/Modified by
% ------------
%     Ilan Ben Hagai, 1-Nov-2010

%%
    m2=-n:n;
    
    epsilon=1.*(m2>=m) + (-1).^(m2-m).*(m2<m);
    mu=abs(m-m2);
    nu=abs(m+m2);
    s=n-(mu+nu)/2;
    Ps=zeros(1,numel(mu));
    for mIdx=1:numel(mu)
        curMu=mu(mIdx);
        curNu=nu(mIdx);
        curS=s(mIdx);
        polyCoeffs=orth_poly('Jacobi',curS,curMu,curNu);
        Ps(mIdx)= polyval(polyCoeffs,cos(beta));
    end
    
    % calculate the Wigner-d function (eq.12)
    d=epsilon.*sqrt(factorial(s).*factorial(s+mu+nu)./(factorial(s+mu).*factorial(s+nu))).*sin(beta/2).^mu.*cos(beta/2).^nu.*Ps;

    % Calculate the coefficients (eq.11) :
    D=exp(-1i*m*alpha-1i*m2*gamma).*d;
end

function pn=orth_poly(class,n,alpha,beta)
% generates an orthogonal polynomial

    if (nargin<4)||isempty(beta)
        beta=0;
    end
    if (nargin<3)||isempty(alpha);
        alpha=0;
    end

    % initialize (-1)'th and zero'th order polynomials
    pn=[];
    pnp1=1;

    for i=0:n
        pnm1=pn;
        pn=pnp1;
        switch class
            case 'Legendre'
                pnp1=((2*i+1)*[pn,0] - i*[0,0,pnm1])/(i+1);
            case 'Hermite'
                pnp1=2*[pn,0] - 2*i*[0,0,pnm1];
            case 'Laguerre'
                pnp1=((2*i+alpha+1)*[0,pn] -[pn,0] - (i+alpha)*[0,0,pnm1])/(i+1);
            case 'Jacobi'
                if (alpha~=0)||(beta~=0)
                    a1n=2*(i+1)*(i+alpha+beta+1)*(2*i+alpha+beta);
                    a2n=(2*i+alpha+beta+1)*(alpha^2-beta^2);
                    if (2*i+alpha+beta)<=150
                        a3n=gamma(2*i+alpha+beta+3)./gamma(2*i+alpha+beta);
                    else
                        a3n=exp(gammaln(2*i+alpha+beta+3)-gammaln(2*i+alpha+beta));
                    end
                    a4n=2*(i+alpha)*(i+beta)*(2*i+alpha+beta+2);
                    pnp1=(a2n*[0,pn] + a3n*[pn,0] - a4n*[0,0,pnm1])./a1n;
                else
                    pnp1=((2*i+1)*[pn,0] - i*[0,0,pnm1])/(i+1);
                end
        end

    end
end

function [ Perm ] = tildize( N )
%A_TILD Summary of this function goes here
%   Detailed explanation goes here
    Perm=(-1).^(2:(N+1)^2+1);
    Perm=diag(Perm);
    for n=0:N;
        Perm(n^2+1:n^2+2*n+1,n^2+1:n^2+2*n+1)=fliplr(Perm(n^2+1:n^2+2*n+1,n^2+1:n^2+2*n+1));
    end
end


function out=meanRespN(N, kr, timeDep, useSOFiA)
% function out=meanRespN(N, kr)
%
% Calculate averaged magnitude pressure at left and right ears
% due to diffraction around a rigid sphere, for order N


  bn=radialMatrix (N,kr,1,inf,timeDep,useSOFiA);
  Yl = shMatrix(N,pi/2,pi/2);
  
  
  tmp=(abs(bn).^2)*(abs(Yl).^2);
  out=sqrt(tmp/(4*pi));
  

end