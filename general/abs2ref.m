function reflection_coeff = abs2ref(absorption_coeff)
%ABS2REF converts absorption coefficients to reflection coefficients (Room Acoustics)

% Author: Ran Weissman, ACLab BGU, 2019

if any(absorption_coeff<0 | absorption_coeff>1)
    error('Absorption coefficients must be between 0 and 1');
end

reflection_coeff = sqrt(1-absorption_coeff);

end