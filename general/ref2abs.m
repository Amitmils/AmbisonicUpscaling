function absorption_coeff = ref2abs(reflection_coeff)
%REF2ABS converts reflection coefficients to absorption coefficients (Room Acoustics)

% Author: Ran Weissman, ACLab BGU, 2019

if any(reflection_coeff<0 | reflection_coeff>1)
    error('Reflection coefficients must be between 0 and 1');
end

absorption_coeff = 1 - reflection_coeff.^2;

end