function h = surf(z, c, zfunc, cfunc, isComplexSH, varargin)
arguments
    z = [];
    c = [];
    zfunc = @abs
    cfunc = zfunc
    isComplexSH (1,1) logical = true
end
% Author: Tom Shlomo, ACLab BGU, 2020

arguments (Repeating)
    varargin
end
if isempty(c) && isempty(z)
    error("you must provide either z or c");
end
nth = 61;
nph = 121;
th = linspace(pi, 0, nth)';
ph = linspace(-pi, pi-10*eps(pi), nph);
th = repmat(th, 1, nph);
ph = repmat(ph, nth, 1);
[x,y] = hammer.project(th, ph);

z = get_samples(z);
c = get_samples(c);
if isempty(c)
    h = surf( x, y, zfunc(z), cfunc(z), varargin{:});
elseif isempty(z)
    h = pcolor( x, y, cfunc(c), varargin{:});
end
shading(h.Parent, 'flat');
hammer.axes(h.Parent);

    function f = get_samples(f)
        if isempty(f)
            return
        end
        if isnumeric(f)
            N = ceil(sqrt(size(f,1))-1);
            f = shmat(N, [th(:) ph(:)], isComplexSH, false)*f;
        elseif isa(f, 'function_handle')
            f = f(th(:), ph(:));
        end
        if size(f,2)>1
            f = sum(abssq(f),2);
        end
        f = reshape(f, nth, nph);
    end
end
