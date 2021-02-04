
function bobj = earo_binauralt_sim(hobj, N, roomDims, recPos, srcAngPos, srcType, freqAbs, alphaAbs, fs, simComponents)

    if nargin < 10
        simComponents = [true true true];  % Simulate [Direct, Spec, Diff]
    end
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
    Receivers=AddReceiver('Location',recPos,'Type','sphharm','MaxOrder',N,'NFComp',true);
    [SrcX SrcY SrcZ] = sph2cart(srcAz,pi/2-srcEl,srcDist);
    Sources=AddSource([],'Location',[SrcX, SrcY, SrcZ] + recPos,'Type',srcType,'Orientation',[radtodeg(pi+srcAz),0,0]);

    Options=MCRoomSimOptions('SimDirect',simComponents(1),'SimSpec',simComponents(2),'SimDiff',simComponents(3),'Fs',double(fs));
    if ~hobj.shutUp, PlotSimSetup(Sources,Receivers,Room); end;

    % Execute and convert SH types
    irArray=RunMCRoomSim(Sources,Receivers,Room,Options);
    P=MCRoomPerm(N)';
    C=SHc2r(N)';
    anm=C*P*irArray.';
    
    % FFT stuff
    anm = fft(anm, [], 2);
    nFFT = size(anm,2);
    fVec = linspace(0,double(fs),nFFT);
    if strcmp(hobj.dataDomain{1},'FREQ'), hobj=hobj.toTime(); end;
    hobj = hobj.toFreq(nFFT);   

    % Trim negative frequencies 
    hobj.data = hobj.data(:,1:(nFFT/2)+1,:);
    anm = anm(:,1:(nFFT/2)+1,:);
    fVec = fVec(1:(nFFT/2)+1);

    % SH transform
    hobj = hobj.toSH(N,'SRC');
    
    
    
    % Iterate through head rotation angles
    if ~hobj.shutUp, fprintf('Generating BRIRs...\n'); end;
    anm_tilde = anm.'*tildize(N);
    brirAngles=(pi/180)*(0:1:359);
    for jj=1:length(brirAngles)

        if ~hobj.shutUp       
            fprintf('   --->Head rotation = %d \r',brirAngles(jj)*180/pi); 
            
        end
        
        % Rotation matrix
        %D=WignerDM(N,0,0,brirAngles(jj));
        D=WignerDM(N,pi/2,-pi/2,-pi/2+brirAngles(jj));
        Hnm_lt_rot=(hobj.data(:,:,1).'*D).';
        Hnm_rt_rot=(hobj.data(:,:,2).'*D).';   
        
        % Generate BRIR
        clear plk prk;
        for ii=1:size(anm_tilde,1)
            plk(ii)=anm_tilde(ii,:)*Hnm_lt_rot(:,ii);
            prk(ii)=anm_tilde(ii,:)*Hnm_rt_rot(:,ii);
        end
        plk(1)=real(plk(1));
        prk(1)=real(prk(1));
        plk(end)=real(plk(end));
        prk(end)=real(prk(end));
        plk=[plk,fliplr(conj(plk(2:end-1)))];
        prk=[prk,fliplr(conj(prk(2:end-1)))];
        arrLt(:,jj)=ifft(plk,'symmetric'); 
        arrRt(:,jj)=ifft(prk,'symmetric');         
    end
    
    newSrir = earo();
    newSrir.name = sprintf('Modeled BRIR in room of dimensions %s',num2str(roomDims));
    newSrir.context = sprintf('Rendered BRIR from ISM Model + HRTF Set; N=%d.',N);
    newSrir.location = 'Virtual room';
    newSrir.date = date;
    newSrir.engineer = 'Jonathan Sheaffer';
    newSrir.contact = 'mail sheaffer@ee.bgu.ac.il';
    newSrir.comments = sprintf('HRTF data is based on %s',hobj.name);
    newSrir.earoVersion = hobj.earoVersion;
    newSrir.type = 'BRIR';
    newSrir.fs = double(hobj.fs);
    newSrir.nData = 360;
    newSrir.capturingSystem = 'binauralt_sim.m';
    newSrir.sourceGrid.r = srcAngPos(1);
    newSrir.positionReference = 'Head Rotation';
    newSrir.micGrid.quadType = 'Gauss-Leg. 360SP (1E/360A)';
    newSrir.scatterer = 1;
    newSrir.micGrid.r = 0.0875;
    newSrir.micGrid.azimuth = brirAngles;
    newSrir.micGrid.quadWeight = ones(1,360)*1/360;
    newSrir.micGrid.elevation = ones(1,360)*pi/2; % 90 elevation
    newSrir.data(:,:,1) = arrLt.';
    newSrir.data(:,:,2) = arrRt.';
    newSrir.orderN = N;
    newSrir.dataDesc = 'BRIR Data head rotation x nData x receivers (1=left ear)';
    newSrir.angles = 'RAD';
    newSrir.dataDomain{1} = 'TIME';
    newSrir.dataDomain{2} = 'SPACE';

    bobj=newSrir;    
    
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