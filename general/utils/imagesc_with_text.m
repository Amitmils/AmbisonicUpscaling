function [h_img, h_txt] = imagesc_with_text(C, opts)
arguments
    C (:,:) double
    opts.x (:,1) double
    opts.y (:,1) double
    opts.text = C+""
    opts.imagesc_additional_args cell = {}
    opts.text_additional_args cell = {}
end

rows = size(C,1);
cols = size(C,2);
if ~isfield(opts, "x")
    opts.x = 1:cols;
end
if ~isfield(opts, "y")
    opts.y = 1:rows;
end


h_img = imagesc(opts.x, opts.y, C, opts.imagesc_additional_args{:});
if isfunc(opts.text)
    num2txt = opts.text;
    opts.text = strings(cols, rows);
    for row=1:rows
        for col=1:cols
            opts.text(row, col) = num2txt( C(row, col) );
        end
    end
end
[y_text, x_text] = ndgrid(opts.y, opts.x);

ax = h_img.Parent;
hold(ax, 'on');
h_txt = text(x_text(:), y_text(:), opts.text(:), ...
    "VerticalAlignment", "middle", ...
    "HorizontalAlignment", "center", ...
    opts.text_additional_args{:});

end

