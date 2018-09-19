function [all_neg_pos, all_neg_size, siamese_size] = neighbor_crop(im, targetPosition, targetSize, neg_siamese_size_flag, pos_ratio, permit_ratio, p, s_z)

all_neg_pos = [];
all_neg_size = [];

sz = [round(s_z) round(s_z)];
im_sz = size(im);
c = (sz+1) / 2;


%check out-of-bounds coordinates, and set them to black
context_xmin = round(targetPosition(2) - c(2)); % floor(pos(2) - sz(2)/2);
context_xmax = context_xmin + sz(2) - 1;
context_ymin = round(targetPosition(1) - c(1)); % floor(pos(1) - sz(1)/2);
context_ymax = context_ymin + sz(1) - 1;
left_pad = max(0, 1-context_xmin);
top_pad = max(0, 1-context_ymin);
right_pad = max(0, context_xmax - im_sz(2));
bottom_pad = max(0, context_ymax - im_sz(1));

context_xmin = context_xmin + left_pad;
context_xmax = context_xmax + left_pad;
context_ymin = context_ymin + top_pad;
context_ymax = context_ymax + top_pad;


siamese_size = [context_ymax - context_ymin + 1, context_xmax - context_xmin + 1];
if neg_siamese_size_flag == 1
    cover_size = pos_ratio * siamese_size;
elseif neg_siamese_size_flag == 2
    cover_size = pos_ratio * targetSize;
end
%         im_neg = im_change(im, targetPosition, crop_size, 'negative', uint8([0, 0, 0]), pos_ratio, 1);
permit_row = [1 - permit_ratio * floor(cover_size(1)/2), size(im, 1) + permit_ratio * floor(cover_size(2)/2)];
permit_col = [1 - permit_ratio * floor(cover_size(1)/2), size(im, 2) + permit_ratio * floor(cover_size(2)/2)];
%         bbox = pos2bbox(targetPosition, [max_size, max_size]);

% all combination
% case_row = [max(-30, -ceil(size(im, 1)./cover_size(1))-2):min(30, ceil(size(im, 1)./cover_size(1))+2)]';
% case_col = [max(-30, -ceil(size(im, 2)./cover_size(2))-2):min(30, ceil(size(im, 2)./cover_size(2))+2)]';
case_row = [max(-p.num_max_negative_search, -ceil(targetPosition(1)./cover_size(1))-2):min(p.num_max_negative_search, ceil((size(im, 1)-targetPosition(1))./cover_size(1))+2)]';
case_col = [max(-p.num_max_negative_search, -ceil(targetPosition(2)./cover_size(2))-2):min(p.num_max_negative_search, ceil((size(im, 2)-targetPosition(2))./cover_size(2))+2)]';
case_col = repmat(case_col, [1, numel(case_row)]);
case_row = repmat(case_row, [size(case_col, 1), 1]);
case_col = case_col';
case_col = case_col(:);

case_neighbor = [case_row, case_col];
case_neighbor(find(sum(case_neighbor == [0, 0], 2) == 2), :) = []; % pos 

trans_val = case_neighbor.*repmat(cover_size, [size(case_neighbor, 1), 1]);
new_pos = repmat(targetPosition, [size(case_neighbor, 1), 1]) + trans_val;
valid_pos = (new_pos(:, 1) >= permit_row(1)) & (new_pos(:, 1) <= permit_row(2)) & (new_pos(:, 2) >= permit_col(1)) & (new_pos(:, 2) <= permit_col(2));
new_pos(find(valid_pos == 0), :) = [];
all_neg_pos = cat(1, all_neg_pos, new_pos);
all_neg_size = cat(1, all_neg_size, repmat(targetSize, [size(new_pos, 1), 1]));
    
if p.visualization
    figure;
    imshow(im/255);
    rect_pos = [targetPosition([2, 1]) - (targetSize([2,1]) - 1)/2, targetSize([2,1])];
    rect_pos2 = [targetPosition([2, 1]) - (siamese_size([2,1]) - 1)/2, siamese_size([2,1])];
    if sum(isnan(rect_pos)) == 0, rectangle('Position',rect_pos, 'EdgeColor','b', 'LineWidth', 4, 'LineStyle', '-'); end
    if sum(isnan(rect_pos2)) == 0, rectangle('Position',rect_pos2, 'EdgeColor','g', 'LineWidth', 2, 'LineStyle', '--'); end
    
    for i = 1 : size(all_neg_pos, 1)
        rect_neg = [all_neg_pos(i, [2, 1]) - (all_neg_size(i, [2,1]) - 1)/2, all_neg_size(i, [2,1])];
        rect_neg2 = [all_neg_pos(i, [2, 1]) - (siamese_size([2,1]) - 1)/2, siamese_size([2,1])];
        if sum(isnan(rect_neg)) == 0, rectangle('Position',rect_neg, 'EdgeColor','r', 'LineWidth', 3, 'LineStyle', '-'); end
        if sum(isnan(rect_neg2)) == 0, rectangle('Position',rect_neg2, 'EdgeColor','y', 'LineWidth', 1, 'LineStyle', '--'); end
    end
end



end
