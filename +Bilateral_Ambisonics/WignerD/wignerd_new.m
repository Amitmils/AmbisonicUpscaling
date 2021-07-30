function D=wignerd_new(n,m,alpha,beta,gamma)
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

%
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
    D=exp(-j*m*alpha-j*m2*gamma).*d;
end
