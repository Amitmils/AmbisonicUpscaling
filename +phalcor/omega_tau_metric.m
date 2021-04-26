function d = omega_tau_metric(omega1, tau1, omega2, tau2, omegaScale, tauScale, opts)
% Author: Tom Shlomo, ACLab BGU, 2020

arguments
    omega1
    tau1
    omega2
    tau2
    omegaScale
    tauScale
    opts.isCart = false
    opts.sqrtFlag = true
    opts.epsilon = inf
    opts.p = 2
end
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here

if ~opts.isCart
    omega1 = s2c(omega1);
    omega2 = s2c(omega2);
end

dtau = abs(tau1 - tau2.')/tauScale;
if isfinite(opts.epsilon)
    n = size(omega1,1);
    m = size(omega2,1);
    d = inf(n,m);
    I = dtau <= opts.epsilon;
    if n>=m
        for i=1:m
            I2 = find(I(:,i));
            domega = real(acos( omega1(I(:,i),:)*omega2(i,:).' ))/omegaScale;
            J = domega <= opts.epsilon;

            I3 = I2(J);
            d_col = domega(J).^opts.p + dtau(I3,i).^opts.p;

            if opts.sqrtFlag
                d_col = (d_col).^(1/opts.p);
            end
            d(I3, i) = d_col;
        end
    else
        for i=1:n
            I2 = find(I(i,:));
            domega = real(acos( omega2(I(i,:),:)*omega1(i,:).' ))'/omegaScale;
            J = domega <= opts.epsilon;

            I3 = I2(J);
            d_col = domega(J).^opts.p + dtau(i,I3).^opts.p;

            if opts.sqrtFlag
                d_col = (d_col).^(1/opts.p);
            end
            d(i,I3) = d_col;
        end
    end
else
    domega = acos(omega1*omega2');
    d = (domega/omegaScale).^opts.p + (dtau).^opts.p;
    
    if opts.sqrtFlag
        d = d.^(1/opts.p);
    end
end

end

