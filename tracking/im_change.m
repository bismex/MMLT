function im_out = im_change(im, pos, sz, flag_method, color3, sz_ratio, flag_on, p)

% color3 = [128, 128, 128];
if flag_on
    sz = sz_ratio*sz;
    rect_out = pos2bbox(pos, sz);

    if strcmp(flag_method, 'positive') % positive
        if numel(color3) > 1
            if ~isempty(p.gpus)
                im_out = gpuArray(single(repmat(reshape(color3, [1, 1, 3]), [size(im, 1), size(im, 2), 1])));
            else
                im_out = single(repmat(reshape(color3, [1, 1, 3]), [size(im, 1), size(im, 2), 1]));
            end
            im_out(max(1, rect_out(1)):min(size(im, 1), rect_out(2)), max(1, rect_out(3)):min(size(im, 2), rect_out(4)), :) = im(max(1, rect_out(1)):min(size(im, 1), rect_out(2)), max(1, rect_out(3)):min(size(im, 2), rect_out(4)), :);
        else
            if ~isempty(p.gpus)
                im_out = gpuArray(single(repmat(color3, [size(im, 1), size(im, 2)])));
            else
                im_out = single(repmat(color3, [size(im, 1), size(im, 2)]));
            end
            im_out(max(1, rect_out(1)):min(size(im, 1), rect_out(2)), max(1, rect_out(3)):min(size(im, 2), rect_out(4))) = im(max(1, rect_out(1)):min(size(im, 1), rect_out(2)), max(1, rect_out(3)):min(size(im, 2), rect_out(4)));
        end
    elseif strcmp(flag_method, 'negative')
        if numel(color3) > 1
            im_out = im;
            im_out(max(1, rect_out(1)):min(size(im, 1), rect_out(2)), max(1, rect_out(3)):min(size(im, 2), rect_out(4)), 1) = color3(1);
            im_out(max(1, rect_out(1)):min(size(im, 1), rect_out(2)), max(1, rect_out(3)):min(size(im, 2), rect_out(4)), 2) = color3(2);
            im_out(max(1, rect_out(1)):min(size(im, 1), rect_out(2)), max(1, rect_out(3)):min(size(im, 2), rect_out(4)), 3) = color3(3);
        else
            im_out = im;
            im_out(max(1, rect_out(1)):min(size(im, 1), rect_out(2)), max(1, rect_out(3)):min(size(im, 2), rect_out(4)), :) = color3;
        end
    else
        error('.');
    end
else
    im_out = im;
end

end