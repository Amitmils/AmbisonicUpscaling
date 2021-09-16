function [ output ] = if1( cond, caseTrue, caseFalse )

% Author: Tom Shlomo, ACLab BGU, 2020

if cond
    output = caseTrue;
else
    output = caseFalse;
end

end

