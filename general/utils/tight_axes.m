function ax = tight_axes(rows,cols, opts)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
arguments
    rows
    cols
    opts.units = "centimeters"
    opts.parent = gcf();
    opts.spacing_x = 0
    opts.spacing_y = 0
    opts.margin_left = 0
    opts.margin_right = 0
    opts.margin_top = 0
    opts.margin_bottom = 0
end
drawnow();
originalUnits = opts.parent.Units;
opts.parent.Units = opts.units;
height = ( opts.parent.Position(4) - opts.margin_top  - opts.margin_bottom - (rows-1)*opts.spacing_y )/rows;
width =  ( opts.parent.Position(3) - opts.margin_left - opts.margin_right  - (cols-1)*opts.spacing_x )/cols;

x = opts.margin_left +   (0:cols-1)*(width  + opts.spacing_x);
y = opts.margin_bottom + (0:rows-1)*(height + opts.spacing_y);
for r=1:rows
    for c=1:cols
        ax(r,c) = axes("Units", opts.units, "Position", [x(c), y(r), width, height], "Parent", opts.parent);
    end
end
ax = flipud(ax);
opts.parent.Units = originalUnits;

end

