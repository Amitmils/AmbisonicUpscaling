function reflections = calc_parametric_rir(room_dim, source_pos, array_pos, R, opts)
% calc_parametric_rir returns a table containing a parametric representation of shoe box room 
% impulse responde, using the image method. The source is assumed to be omni-directional.
% reflections is a table, each row (excepts the first) represents an image
% source (a reflection). The first source represents the direct sound.
%   - delay:    The time of propogatin from the source to the array. sec.
%   - omega:    The DOA of the source relative to the array. two columns:
%               [elevcation, azimuth]. radidans.
%   - amp:      The amplitude of the source. Includes both radial attenuation due to distance, and 
%               due to hitting room boundaries.
%   - r:        The distance of the source from the array. meters.
%   - relative_pos: The relative position of the source from the array. [dx, dy, dz] meters.
%   - hits:     A 6 row vector counting the time the reflection hit each wall.
%               Examples:
%                   [0, 0, 0, 0, 0, 0] - The direct sound.
%                   [0, 0, 1, 0, 0, 0] - The first reflection for the floor.
%                   [0, 0, 0, 0, 0, 1] - The first reflection from the ceiling.
%                   [1, 0, 0, 2, 0, 0] - A reflection that hits the left wall once and the right 
%                                        wall twice.
%
% Author: Tom Shlomo, ACLab BGU, 2020

arguments
    room_dim (1,3) double
    % Dimensions of the room, meters. [Lx, Ly, Lz].
    
    source_pos (1,3) double
    % The location of the source inside the room, meters [x, y, z].
    
    array_pos (1,3) double
    % The location of the array inside the room, meters [x, y, z].
    
    R (1,:) double
    % Walls reflection coeffecients. Either a scalar, or a 6-vector so that each wall has a different
    % coeffecient according to the following convention:
    % [x=0, y=0, z=0, x=Lx, y=Ly, z=Lz]

    % Name-Value pairs:
    opts.t_max (1,1) double = inf         
    % The maximal delay. Default value is inf
    
    opts.amp_thresh (1,1) double = 1e-4
    % Any reflection whos amplitude is lower then the amplitude of the direct*AmpThresh, will be
    % discarded. Default is 1e-4.
    
    opts.max_reflection_order (1,1) double = 200
    % Reflections are calculated only up to this order.
    % Can be either a scalar, or a 3 elements row vector (x,y,z).
    % Default is 200 (this is very large usually).
    % Too large value may lead to memmory issues.
    
    opts.energy_thresh (1,1) double = 0
    % If positive, than the late part of the response will be removed. The energy of that part is
    % approximately EnergyThresh times the total energy.
    % Default if 0 (so nothing is removed according to this test).
    
    opts.max_waves (1,1) double = inf  
    % The maximum number of waves in the response. If the number of waves is larger, then late 
    % reflections will be removed. Default is inf.
    
    opts.c (1,1) double = soundspeed()              
    % Speed of sound, in m/sec. Default is the output of the function soundspeed().
    
    opts.zero_first_delay (1,1) logical = false
    % If true, the delays are shifted so that the first is 0. Default is false.
    
    opts.angle_dependence (1,1) logical = true
    % If true, the walls reflection coeffecients depend on the angle of incident. For more details,
    % see calc_reflection_amp.
end

%% input validation
if isscalar(R)
    R = repmat(R,1,6);
end
assert(isvector(R) && length(R)==6, "R must be a scalar of a 6 vector");

if isscalar(opts.max_reflection_order)
    opts.max_reflection_order = repmat(opts.max_reflection_order,1,3);
end
max_reflection_order_by_Tmax = ceil(opts.t_max *opts.c./room_dim)+1;
max_reflection_order_by_amp_thresh = inf(1,3);
if ~isempty(opts.amp_thresh)
    opts.amp_thresh = opts.amp_thresh / norm( source_pos-array_pos );
    L_diag = norm(room_dim);
    R_geomean = sqrt(R(1:3).*R(4:6));
    for i=1:3
        if R_geomean(i)==0
            max_reflection_order_by_amp_thresh(i) = 1;
        else
            a = @(n) R_geomean(i).^n ./ (n.*room_dim(i) + L_diag) - opts.amp_thresh;
            int = [0 log(opts.amp_thresh)/log(R_geomean(i))];
            max_reflection_order_by_amp_thresh(i) = ceil( fzero( a, int, struct("TolX", 0.5) )/2 )*2;
        end
    end
end

max_reflection_order = min([opts.max_reflection_order; max_reflection_order_by_amp_thresh; max_reflection_order_by_Tmax], [], 1);

%% calculate all parametric data except amplitudes
reflections = table();
[reflections_pos, reflections.hits] = image_method.calc_reflections_info(room_dim, source_pos, R, max_reflection_order);
reflections.relative_pos = reflections_pos - array_pos;
reflections.r = vecnorm(reflections.relative_pos,2,2);
reflections.delay = reflections.r/opts.c;

%% filter by opts.t_max 
I = reflections.delay<=opts.t_max;
reflections = reflections(I,:);

%% calculate amplitudes (comes after opts.t_max filtering for better performance)
reflections.amp = image_method.calc_reflection_amp(reflections.relative_pos, reflections.hits, (R+1)./(1-R), opts.angle_dependence, reflections.r);

%% filter by opts.amp_thresh
I = abs(reflections.amp) >= opts.amp_thresh;
reflections = reflections(I,:);

%% sort by delay
reflections = sortrows(reflections, 'delay');

%% filter by opts.energy_thresh
if opts.energy_thresh>0
    accumulated_energy = cumsum(abssq(reflections.amp));
    k = find( accumulated_energy > (1-opts.energy_thresh)*accumulated_energy(end), 1 );
    if ~isempty(k)
        reflections(k+1:end,:) = [];
    end
end

%% filter by max waves
reflections(opts.max_waves+1:end,:) = [];

%% zero first delay
if opts.zero_first_delay
    reflections.delay = reflections.delay - reflections.delay(1);
end

%% calculate omega
[reflections.omega(:,1), reflections.omega(:,2)] = c2s(reflections.relative_pos);

end

