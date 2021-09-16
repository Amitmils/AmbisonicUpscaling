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
