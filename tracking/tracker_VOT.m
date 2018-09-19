function tracker_VOT(varargin)

p = varargin{1};
tic;

%% Intialization at the first frame

if strcmp(p.type, 'vot')
    cleanup = onCleanup(@() exit() ); % Tell Matlab to exit once the function exits
    RandStream.setGlobalStream(RandStream('mt19937ar', 'Seed', sum(clock))); % Set random seed to a different value every time as required by the VOT rules
    [handle, im_file, region] = vot('rectangle');
    [cx, cy, w, h] = get_axis_aligned_BB(region);
    region = [cx, cy, w, h];
    targetPosition = region([2, 1]);
    targetSize = region([4, 3]);
    im = imread(im_file);
    im = single(im);
elseif strcmp(p.type, 'demo')
    [~, targetPosition, targetSize, img_files] = load_video_info(p.seq_base_path, p.video);
    im = imread(img_files{p.startFrame});
    im = single(im);
end

if ~isempty(p.gpus)
    reset(p.gpu_device);
    wait(p.gpu_device);
end

net_z = load_pretrained([p.net_base_path p.net], p.gpus);
net_x = load_pretrained([p.net_base_path p.net], p.gpus);

% To solve the GPU memory error
if size(im, 1) * size(im, 2) > p.num_limit_pixel
    num_resize = ceil(sqrt(size(im, 1) * size(im, 2)/p.num_limit_pixel)) + p.gpu_memory_resize_add;
else
    num_resize = p.gpu_memory_resize_basic;
end

oTargetPosition = targetPosition;
oTargetSize = targetSize;
confidence = 1;

if ~isempty(p.gpus)
    im = gpuArray(im);
end
if(size(im, 3)==1)
    im = repmat(im, [1 1 3]);
end

stats = [];

% exemplar branch (used only once per video) computes features for the target
remove_layers_from_prefix(net_z, p.prefix_x);
remove_layers_from_prefix(net_z, p.prefix_join);
remove_layers_from_prefix(net_z, p.prefix_adj);
% instance branch computes features for search region x and cross-correlates with z features
remove_layers_from_prefix(net_x, p.prefix_z);
zFeatId = net_z.getVarIndex(p.id_feat_z);
scoreId = net_x.getVarIndex(p.id_score);
% Init visualization
videoPlayer = [];
if p.visualization && isToolboxAvailable('Computer Vision System Toolbox')
    videoPlayer = vision.VideoPlayer('Position', [100 100 [size(im,2), size(im,1)]+30]);
end
% get avg for padding
avgChans = gather([mean(mean(im(:,:,1))) mean(mean(im(:,:,2))) mean(mean(im(:,:,3)))]);


wc_z = targetSize(2) + p.contextAmount*sum(targetSize);
hc_z = targetSize(1) + p.contextAmount*sum(targetSize);
s_z = sqrt(wc_z*hc_z);
scale_z = p.exemplarSize / s_z;
[z_crop, ~] = get_subwindow_tracking(im, targetPosition, [p.exemplarSize p.exemplarSize], [round(s_z) round(s_z)], avgChans, p);
if p.subMean
    z_crop = bsxfun(@minus, z_crop, reshape(stats.z.rgbMean, [1 1 3]));
end

d_search = (p.instanceSize - p.exemplarSize)/2;
pad = d_search/scale_z;
s_x = s_z + 2*pad;
s_ratio = s_z/s_x;
% arbitrary scale saturation
min_s_x = 0.2*s_x;
max_s_x = 5*s_x;

first_s_x = s_x;

switch p.windowing
    case 'cosine'
        window = single(hann(p.scoreSize*p.responseUp) * hann(p.scoreSize*p.responseUp)');
    case 'uniform'
        window = single(ones(p.scoreSize*p.responseUp, p.scoreSize*p.responseUp));
end
% make the window sum 1
window = window / sum(window(:));
scales = (p.scaleStep .^ ((ceil(p.numScale/2)-p.numScale) : floor(p.numScale/2)));
% evaluate the offline-trained network for exemplar z features

net_z.eval({'exemplar', z_crop});
clear z_crop;
z_features = net_z.vars(zFeatId).value;
z_features_4d = repmat(z_features, [1 1 1 p.numScale]);
net_z.reset();

%% Semantic

% import VGGNet
net = load([p.net_base_path, p.redetection_net]);
net.layers(37:end) = [];
net = dagnn.DagNN.fromSimpleNN(net);
if p.gpus
    net.move('gpu')
end
% Saving semantic feature for redetection
first_s_z = s_z;

[crop_image, ~] = get_subwindow_tracking(im, targetPosition, [round(s_z) round(s_z)], [round(s_z) round(s_z)], avgChans, p);
if num_resize > 1
    net.eval({'x0', imresize(crop_image, 1/num_resize)})
else
    net.eval({'x0', crop_image})
end
crop_feature = net.vars(end).value;
net.reset();
clear crop_image;

if ~isempty(p.gpus)
    wait(p.gpu_device);
end

% fprintf(['Image size : ', num2str(size(im)), '\n']);

if ~isempty(p.gpus)
    wait(p.gpu_device);
end
if num_resize > 1
    im_tmp = imresize(im, 1/num_resize);
    net.eval({'x0', im_tmp})
    clear im_tmp;
else
    net.eval({'x0', im});
end
% entire_image = gpuArray(single(im));
entire_feature = net.vars(end).value;
net.reset();
if ~isempty(p.gpus)
    wait(p.gpu_device);
end

if ~isempty(p.gpus)
    corr_feature = gather(vl_nnconv(entire_feature, crop_feature,[]));
else
    corr_feature = vl_nnconv(entire_feature, crop_feature,[]);
end
clear entire_feature;

if ~isempty(p.gpus)
    wait(p.gpu_device);
end

best_score(1) = max(corr_feature(:));
crop_vector = max(max(crop_feature,[],1),[],2);
crop_vector = crop_vector./(sqrt(sum(crop_vector.^2,3)));
crop_RMAC = RMAC(crop_feature);


% Extract best peak at the first frame (for normalization)
scaledInstance = s_x .* scales;
scaledTarget = [targetSize(1) .* scales; targetSize(2) .* scales];
x_crops = make_scale_pyramid(im, targetPosition, scaledInstance, p.instanceSize, avgChans, stats, p);
[~, ~, bestPeakFirst] = tracker_eval(net_x, round(s_x), scoreId, z_features_4d, x_crops, targetPosition, window, p);

% discard region when redetection
p.cand_region = min(min(size(corr_feature)./[size(im, 1), size(im, 2)]*s_x/p.cand_region), min(size(corr_feature))); % re-detection ÇÒ ¶§ masking
p.cand_region = floor(p.cand_region/2);
first_targetSize = targetSize;
ratio_scale = p.scaleStep.^p.pow_value;
redetect_scales = ratio_scale.^p.pow_scale;
redetect_scales(redetect_scales < p.min_scale_before) = [];
redetect_scales(redetect_scales > p.max_scale_before) = [];
[target_ratio, ~] = max(targetSize./[size(im, 1), size(im, 2)]);
redetect_scales(target_ratio.*redetect_scales<p.min_scale_after) = [];
redetect_scales(target_ratio.*redetect_scales>p.max_scale_after) = [];
while numel(redetect_scales) > p.max_redetection_num
    redetect_scales_tmp = redetect_scales;
    redetect_scales_tmp(redetect_scales_tmp<1) = 1./redetect_scales_tmp(redetect_scales_tmp<1);
    [~, delete_idx] = max(redetect_scales_tmp);
    redetect_scales(delete_idx) = [];
end
if numel(redetect_scales)<=1
    redetect_scales = cat(2, redetect_scales, [0.5, 0.75, 1, 1.25, 1.5]);
    redetect_scales = unique(redetect_scales);
end

redetect_num_scales = numel(redetect_scales);   % the number of candidate for redection


% Initialization
bestPeak = 0;           % siamese best Peak
redetection = 0;        % redetection flag, 0 : tracking success, 1: tracking fail
tracking_sequence = 1;  
short_term_continue = 0;  % short-term tracking
long_store = 0;         % long-term save flag
long_sequence =0 ;      
f = gather(net_x.params(23).value); % last output filter param of siamese network
b = gather(net_x.params(24).value); % bias param value
long_score = 0;         % MAC/R-MAC score
short_fail = 0;

short_memory.feature(:,:,:,1) = gather(z_features); % feature of siamese network
short_memory.weights = reshape( (exp((1:p.short_memory_frames)/p.short_memory_damping)), [1 1 1 p.short_memory_frames]);
long_memory.feature(:,:,:,1) = gather(z_features);  % siamese feature for tracking
long_memory.semantic(:,:,:,1) = gather(crop_feature);   % semantic feature for redection
long_memory.vector(:,1) = gather(crop_vector);  % for MAC vector
long_memory.memory_length = 1;                  % long-term memory length
long_memory.RMAC(:,1) = gather(crop_RMAC);      % R-MAC vector
long_memory.forgetting(1)= 1;
long_memory.time(1) = 0;
long_param.confidence = []; 
clear crop_RMAC;
clear crop_vector;

% Extraction for negative sample
neg_z_features = [];
if p.num_neg_siamese + p.num_neg_cnn > 0
    neg_compare = 1;
else
    neg_compare = 0;
end
if neg_compare
    [all_neg_pos, all_neg_size, siamese_size] = neighbor_crop(im, targetPosition, targetSize, p.neg_siamese_size_flag, p.pos_ratio, p.permit_ratio, p, s_z); % find all combinations
    im_neg = im_change(im, targetPosition, siamese_size, 'negative', uint8(avgChans), 1, 1, p);
    cand_logical = zeros(size(all_neg_pos, 1), 1); % candidate index
    
    if p.num_neg_siamese > 0 % siamese-based
        cand_peak = [];
        for j = 1 : size(cand_logical, 1)
            test_pos = all_neg_pos(j, :);
            x_crops_cand = make_scale_pyramid(im, test_pos, scaledInstance, p.instanceSize, avgChans, stats, p);
            if ~isempty(p.gpus)
                wait(p.gpu_device);
            end
            [~, ~, cand_peak_tmp] = tracker_eval(net_x, round(s_x), scoreId, z_features_4d, x_crops_cand, targetPosition, window, p);
            cand_peak = cat(1, cand_peak, gather(cand_peak_tmp)); % collect bestpeak
            clear x_crops_cand;
            clear cand_peak_tmp;
        end
        [~, good_siamese_cand] = sort(cand_peak, 'descend');
        cand_logical(good_siamese_cand(1:min(numel(good_siamese_cand), p.num_neg_siamese))) = 1;
    end
    for j = 1 : sum(cand_logical) % create the negative image
        idx = find(cand_logical);
        im_neg = im_change(im_neg, all_neg_pos(idx(j), :), siamese_size, 'negative', uint8(avgChans), 1, 1, p);
    end
    
    if p.num_neg_cnn > 0 && mean(cand_logical) ~= 1 % cnn-based
        all_neg_pos_tmp = all_neg_pos;
        all_neg_pos_tmp(find(cand_logical), :) = inf;
        
        
        if ~isempty(p.gpus)
            wait(p.gpu_device);
        end
        if num_resize > 1
            im_tmp = imresize(im_neg, 1/num_resize);
            net.eval({'x0', im_tmp})
            clear im_tmp;
        else
            net.eval({'x0', im_neg});
        end
        if ~isempty(p.gpus)
            wait(p.gpu_device);
        end
        neg_feature = net.vars(end).value;
        net.reset();
        neg_corr_feature = gather(vl_nnconv(neg_feature, crop_feature,[]));
        clear neg_feature;
        if ~isempty(p.gpus)
            wait(p.gpu_device);
        end
        
        for j = 1 : p.num_neg_cnn
%             im_neg_gpu = gpuArray(single(im_neg));
            max_v = max(neg_corr_feature(:));
            [max_r, max_c, ~] = ind2sub(size(neg_corr_feature), find(neg_corr_feature==max_v));
            cand_row = max_r(1); cand_col = max_c(1);
            
            pow_pool = pow2(p.num_pool) * num_resize;
            if p.center_refine_flag
                cand_row = round(pow_pool*(cand_row + size(crop_feature,1)/2) + pow_pool/2);
                cand_col = round(pow_pool*(cand_col + size(crop_feature,2)/2) + pow_pool/2);
            else
                cand_row = pow_pool*(cand_row + floor(size(crop_feature,1)/2) );
                cand_col = pow_pool*(cand_col + floor(size(crop_feature,2)/2) );
            end
            cand_row = repmat(cand_row, [size(all_neg_pos_tmp, 1), 1]);
            cand_col = repmat(cand_col, [size(all_neg_pos_tmp, 1), 1]);
            dist_cand = sum((all_neg_pos_tmp - [cand_row, cand_col]).^2, 2);
            [min_val, min_idx] = min(dist_cand);
            all_neg_pos_tmp(min_idx, :) = inf;
            if isinf(min_val), break; end
            cand_logical(min_idx) = 1; % candidate
            if mean(cand_logical) == 1, break;  end
            if ~p.neg_cnn_first_center
                neg_corr_feature(max(max_r-p.cand_region,1):min(max_r+p.cand_region, size(neg_corr_feature,1)), max(max_c-p.cand_region,1):min(max_c+p.cand_region,size(neg_corr_feature,2))) = 0;
            end
            if sum(neg_corr_feature(:))==0, break; end
        end
    end
    clear crop_feature;
    all_neg_pos = all_neg_pos(find(cand_logical), :);
    all_neg_size  = all_neg_size(find(cand_logical), :);
    
    im_neg = im_change(im, targetPosition, targetSize, 'negative', uint8(avgChans), p.neg_ratio, p.neg_crop_flag, p);
    for i = 1 : size(all_neg_size, 1)
        neg_pos = all_neg_pos(i, :);
        neg_sz = all_neg_size(i, :);
        wc_z_neg = neg_sz(2) + p.contextAmount*sum(neg_sz);
        hc_z_neg = neg_sz(1) + p.contextAmount*sum(neg_sz);
        s_z_neg = sqrt(wc_z_neg*hc_z_neg);
        [z_crop_neg, ~] = get_subwindow_tracking(im_neg, neg_pos, [p.exemplarSize p.exemplarSize], [round(s_z_neg) round(s_z_neg)], avgChans, p);
        if p.subMean
            z_crop_neg = bsxfun(@minus, z_crop_neg, reshape(stats.z.rgbMean, [1 1 3]));
        end
        net_z.eval({'exemplar', z_crop_neg});
        z_features_tmp_neg = net_z.vars(zFeatId).value;
        net_z.reset();
        z_features_neg = repmat(z_features_tmp_neg, [1 1 1 p.numScale]);
        neg_z_features{i}.z_features_neg = z_features_neg;
        clear z_crop_neg;
        clear z_features_neg;
        clear z_features_tmp_neg;
    end
    
    if numel(neg_z_features) == 0, neg_compare = 0; end
end
clear im_neg;


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


if p.store == 1
    f_struct = write_txt([], p.root_name, p.folder_name, p.video, [], [], [], 1); % open (init)
end

% compare with other result
if p.compare_result == 1
    my_result = importdata(['./' p.compare_result_name '/longterm/' p.video '/' p.video '_001.txt']);
    result_confidence = importdata(['./' p.compare_result_name '/longterm/' p.video '/' p.video '_001_confidence.value']);
end







%% start tracking ----------------------------------------------------------------------------------------
if strcmp(p.type, 'vot')
    i = 1;
elseif strcmp(p.type, 'demo')
    i = p.startFrame;
end
while true
    fail_peak = 0;
    if i > 1
        tic;
        clear im;
        if ~isempty(p.gpus)
            wait(p.gpu_device);
        end
        
        if strcmp(p.type, 'vot')
            [handle, im_file] = handle.frame(handle);
            if isempty(im_file), break; end
            im = single(imread(im_file));
        elseif strcmp(p.type, 'demo')
            if numel(img_files) >= i
                im = single(imread(img_files{i}));
            else
                break;
            end
        end
        if ~isempty(p.gpus)
            wait(p.gpu_device);
            im = gpuArray(im);
            if strcmp(p.type, 'vot')
%                 fprintf([num2str(i), ' : ' , num2str(p.gpu_device.AvailableMemory), '\n']);
            elseif strcmp(p.type, 'demo')
%                 fprintf([p.video, ' - ', num2str(i), ' : ' , num2str(p.gpu_device.AvailableMemory), '\n']);
            end
        end
        if(size(im, 3)==1)
            im = repmat(im, [1 1 3]);
        end
        
        % forgetting curve update
        long_memory.time(1) = 0;
        long_memory.forgetting = p.forgetting_peak.*exp(-long_memory.time./(p.forget_h*p.forget_strength));
        
        memory_forget = long_memory.forgetting < p.forgetting_threshold;
        if sum(memory_forget) ~= 0
            long_memory.memory_length =  long_memory.memory_length - 1;
            long_memory.feature(:,:,:,memory_forget) = [];
            long_memory.semantic(:,:,:,memory_forget) = [];
            long_memory.vector(:,memory_forget) = [];
            long_memory.RMAC(:,memory_forget) = [];
            best_score(memory_forget) = [];
            
            p.forget_strength(memory_forget) = [];
            long_memory.time(memory_forget) = [];
            long_memory.forgetting(memory_forget) = [];
            p.forgetting_peak(memory_forget) = [];
        end
        
        % if grayscale repeat one channel to match filters size
        %% if the target position is not nan value in a previous frame
        if ~isnan(targetPosition)
            scaledInstance = s_x .* scales;
            scaledTarget = [targetSize(1) .* scales; targetSize(2) .* scales];
            % extract scaled crops for search region x at previous target position
            clear x_crops;
            x_crops = make_scale_pyramid(im, targetPosition, scaledInstance, p.instanceSize, avgChans, stats, p);
            % evaluate the offline-trained network for exemplar x features
            [newTargetPosition, newScale, bestPeak] = tracker_eval(net_x, round(s_x), scoreId, z_features_4d, x_crops, targetPosition, window, p);
        else
            bestPeak = 0;
        end
        
        %% %if success (tracking success)
        if bestPeak/p.siamese_score_mean > p.siamese_score_threshold && bestPeak/bestPeakFirst > p.peak_ratio_threshold && short_fail < p.fail_th_tracking
            
            targetPosition = gather(newTargetPosition);
            s_x = max(min_s_x, min(max_s_x, (1-p.scaleLR)*s_x + p.scaleLR*scaledInstance(newScale)));
            targetSize = (1-p.scaleLR)*targetSize + p.scaleLR*[scaledTarget(1,newScale) scaledTarget(2,newScale)];
            
            redetection = 0;
            tracking_sequence = tracking_sequence + 1;
            
            % Saving siamese score!, update based on FIFO
            if length(long_param.confidence) == p.siamese_score_num_save
                long_param.confidence(1:p.siamese_score_num_save-1) = long_param.confidence(2:p.siamese_score_num_save);
                long_param.confidence(p.siamese_score_num_save) = gather(bestPeak);
            else
                long_param.confidence(end+1) = gather(bestPeak);
            end
            p.siamese_score_mean = mean(long_param.confidence);
            
            %% short-term memory
            if bestPeak/bestPeakFirst > p.short_term_store_threshold
                short_fail = 0;
                short_term_continue = short_term_continue + 1;
                s_z = s_x*s_ratio;
                
                [short_crop, ~] = get_subwindow_tracking(im, targetPosition, [p.exemplarSize p.exemplarSize], [round(s_z) round(s_z)], avgChans, p);
                net_z.eval({'exemplar', short_crop});
                clear z_features_tmp2;
                z_features_tmp2 = net_z.vars(zFeatId).value;
                net_z.reset();
                
                % short-term memory update
                if size(short_memory.feature,4) == p.short_memory_frames
                    short_memory.feature(:,:,:,2:p.short_memory_frames-1) = gather(short_memory.feature(:,:,:,3:p.short_memory_frames));
                    short_memory.feature(:,:,:,p.short_memory_frames) = gather(z_features_tmp2);
                else
                    short_memory.feature(:,:,:,end+1) = gather(z_features_tmp2);
                end
                
                % short-term filter update (siamese),
                z_features_short = sum(bsxfun(@times, short_memory.feature, short_memory.weights(1,1,1,1:size(short_memory.feature,4))),4)/sum(short_memory.weights(1,1,1,1:size(short_memory.feature,4)),4);
                clear z_features_4d;
                if ~isempty(p.gpus)
                    z_features_long_short = gpuArray(single(((1 - (p.filter_short_long_ratio)*exp(-tracking_sequence/p.short_long_damping))*z_features + ((p.filter_short_long_ratio)*exp(-tracking_sequence/p.short_long_damping))*z_features_short)));
                    z_features_4d = gpuArray(single(repmat(z_features_long_short, [1 1 1 p.numScale])));
                else
                    z_features_long_short = single(((1 - (p.filter_short_long_ratio)*exp(-tracking_sequence/p.short_long_damping))*z_features + ((p.filter_short_long_ratio)*exp(-tracking_sequence/p.short_long_damping))*z_features_short));
                    z_features_4d = single(repmat(z_features_long_short, [1 1 1 p.numScale]));
                end
                clear z_features_short;
                clear z_features_long_short;
                
                %% retrieval(MAC/R-MAC) score for long-term store
                % Computing retrieval score for saving long-term memory 
                % Retrieval score based on VGGNet feature
                s_z = s_x*s_ratio;
                [semantic_crop, ~] = get_subwindow_tracking(im, targetPosition, [round(first_s_z) round(first_s_z)], [round(s_z) round(s_z)], avgChans, p);
                
                if num_resize > 1
                    net.eval({'x0', imresize(semantic_crop, 1/num_resize)})
                else
                    net.eval({'x0', semantic_crop})
                end
                
                clear semantic_crop;
                clear semantic_feature;
                clear semantic_vector;
                clear semantic_RMAC;
                semantic_feature = net.vars(end).value;
                net.reset();
                semantic_RMAC = RMAC(semantic_feature);
                semantic_vector = max(max(semantic_feature,[],1),[],2);
                semantic_vector = semantic_vector./(sqrt(sum(semantic_vector.^2,3)));
                
                RMAC_score = (sum(bsxfun(@times, squeeze(semantic_RMAC), long_memory.RMAC),1));
                %long_memory_weight = sum(bsxfun(@times, squeeze(semantic_vector), long_memory.vector),1);
                
                % forgetting
                long_select = RMAC_score > p.retrieval_threshold;
                p.forgetting_peak(long_select) = min(1, p.forgetting_up *long_memory.forgetting(long_select));
                long_memory.time(long_select) = 0;
                long_memory.forgetting(long_select) = p.forgetting_peak(long_select).*exp(-long_memory.time(long_select)/(p.forget_h*p.forget_strength(long_select)));
                
                long_score = mean(RMAC_score);
                
                if long_score > p.retrieval_threshold
                    long_sequence = long_sequence + 1;
                else
                    long_sequence = 0;
                end
                if long_sequence >= p.long_memory_interval
                    if neg_compare
                        
                        if p.compare_crop_flag
                            im_pos = im_change(im, targetPosition, targetSize, 'negative', uint8(avgChans), p.compare_crop_ratio, 1, p);
                            clear x_crops;
                            x_crops = make_scale_pyramid(im_pos, targetPosition, scaledInstance, p.instanceSize, avgChans, stats, p);
                        end
                        
                        all_neg_pos = [];
                        all_neg_bestPeak = [];
                        for n = 1 : numel(neg_z_features)
                            [neg_pos, new_scale, new_bestPeak] = tracker_eval(net_x, round(s_x), scoreId, neg_z_features{n}.z_features_neg, x_crops, targetPosition, window, p);
                            neg_pos = gather(neg_pos);
                            all_neg_pos = cat(1, all_neg_pos, neg_pos);
                            all_neg_bestPeak = cat(1, all_neg_bestPeak, gather(new_bestPeak));
                        end
                        
                        if p.neg_in_box_flag
                            max_pos = targetPosition + (targetSize - 1) / 2;
                            min_pos = targetPosition - (targetSize - 1) / 2;
                            max_pos = repmat(max_pos, [size(all_neg_pos, 1), 1]);
                            min_pos = repmat(min_pos, [size(all_neg_pos, 1), 1]);
                            delete_idx = find(sum((all_neg_pos <= max_pos) & (all_neg_pos >= min_pos), 2) ~= 2);
                            all_neg_pos(delete_idx, :) = [];
                            all_neg_bestPeak(delete_idx, :) = [];
                        end
                        
                        pos_bestPeak = gather(bestPeak);
                        if ~isempty(all_neg_pos)
                            if sum(p.peak_ratio_short * pos_bestPeak < all_neg_bestPeak) == 0
                                long_store = 1;
                                long_sequence = 0;
                            else
                                long_store = 0;
                                if p.long_cnt_restart
                                    long_sequence = 0;
                                end
                            end
                        else
                            long_store = 1;
                            long_sequence = 0;
                        end
                    else
                        long_store = 1;
                        long_sequence = 0;
                    end
                else
                    long_store = 0;
                end
                
                %% long-term memory
                if long_store == 1
                    short_term_continue = 0;
                    long_store = 0;
                    long_sequence = 0;
                    
                    if ~isempty(p.gpus)
                        wait(p.gpu_device);
                    end

                    if num_resize > 1
                        im_tmp = imresize(im, 1/num_resize);
                        net.eval({'x0', im_tmp})
                        clear im_tmp;
                    else
                        net.eval({'x0', im})
                    end
                    entire_feature = net.vars(end).value;
                    net.reset();
                    
                    if ~isempty(p.gpus)
                        wait(p.gpu_device);
                    end
                    
                    corr_feature = gather(vl_nnconv(entire_feature, semantic_feature,[]));
                    clear entire_feature;
                    
                    if ~isempty(p.gpus)
                        wait(p.gpu_device);
                    end
                    if long_memory.memory_length == p.long_memory_frames
                        long_forget = min(long_memory.forgetting,0.95);
                        long_forget(1) = 1;
                        forget_index = find(long_forget== min(long_forget));
                        forget_index = forget_index(1);
                        long_memory.feature(:,:,:,forget_index) = [];
                        long_memory.semantic(:,:,:,forget_index) = [];
                        long_memory.vector(:,forget_index) = [];
                        long_memory.RMAC(:,forget_index) = [];
                        best_score(forget_index) = [];
                        
                        p.forget_strength(forget_index) = [];
                        long_memory.time(forget_index) = [];
                        long_memory.forgetting(forget_index) = [];
                        p.forgetting_peak(forget_index) = [];
                        
                        long_memory.feature(:,:,:,end+1) = gather(z_features_tmp2);
                        long_memory.semantic(:,:,:,end+1) = gather(semantic_feature);
                        long_memory.vector(:,end+1) = gather(semantic_vector);
                        long_memory.RMAC(:,end+1) = gather(semantic_RMAC);
                        best_score(end+1) = max(corr_feature(:));
                        
                        p.forget_strength(end+1) = gather(long_score);
                        long_memory.time(end+1) = 0;
                        long_memory.forgetting(end+1) = 1;
                        p.forgetting_peak(end+1) = 1;
                    else
                        
                        long_memory.memory_length =  long_memory.memory_length + 1;
                        long_memory.feature(:,:,:,end+1) = gather(z_features_tmp2);
                        long_memory.semantic(:,:,:,end+1) = gather(semantic_feature);
                        long_memory.vector(:,end+1) = gather(semantic_vector);
                        long_memory.RMAC(:,end+1) = gather(semantic_RMAC);
                        best_score(end+1) = max(corr_feature(:));
                        
                        p.forget_strength(end+1) = gather(long_score);
                        long_memory.time(end+1) = 0;
                        long_memory.forgetting(end+1) = 1;
                        p.forgetting_peak(end+1) = 1;
                    end
                    
                    % update
                    clear z_features;
                    z_features = (0.7 + 0.3*exp(-long_memory.memory_length/10))*long_memory.feature(:,:,:,1) + ...
                        (0.3 - 0.3*exp(-long_memory.memory_length/10))*sum(long_memory.feature(:,:,:,2:end), 4)./(long_memory.memory_length-1);
                    
                    % bestPeak update about long-term filter
                    tmp = sum(sum(sum(z_features.^2,1),2),3);
                    bestPeakFirst = gather(tmp*f + b);
                end
            else
                % for confidence allocation
                short_fail = short_fail + 1;
                short_term_continue = 0;
                long_sequence = 0;
                s_z = s_x*s_ratio;

                % make semantic input
                [semantic_crop, ~] = get_subwindow_tracking(im, targetPosition, [round(first_s_z) round(first_s_z)], [round(s_z) round(s_z)], avgChans, p);
                if num_resize > 1
                    net.eval({'x0', imresize(semantic_crop, 1/num_resize)})
                else
                    net.eval({'x0', semantic_crop})
                end
                clear semantic_crop;
                clear semantic_feature;
                clear semantic_vector;
                clear semantic_RMAC;
                semantic_feature = net.vars(end).value;
                net.reset();
                semantic_RMAC = RMAC(semantic_feature); % R-MAC vector
                semantic_vector = max(max(semantic_feature,[],1),[],2);
                semantic_vector = semantic_vector./(sqrt(sum(semantic_vector.^2,3)));   % MAC  vector
                
                % for confidence
                RMAC_score = mean(sum(bsxfun(@times, squeeze(semantic_RMAC), long_memory.RMAC),1));
                long_score = RMAC_score;
                
            end
        else
            %% if not(tracking fail)- re-detection
            if ~isempty(p.gpus)
                wait(p.gpu_device);
            end
            % Initialize short-term memory
            short_memory.feature=[]; short_memory.feature(:,:,:,1) = gather(z_features);
            clear z_features_4d;
            if ~isempty(p.gpus)
                z_features_4d = gpuArray(single(repmat(z_features, [1 1 1 p.numScale])));
            else
                z_features_4d = single(repmat(z_features, [1 1 1 p.numScale]));
            end
            redetection = 1;
            tracking_sequence = 1;
            short_term_continue = 0; long_sequence = 0;
            if p.fail_flag
                short_fail = p.fail_th_tracking - p.fail_th_redetection;
            else
                short_fail = 0;
            end
            
            if ~isempty(p.gpus)
                wait(p.gpu_device);
            end
            % re-detection (entire image)
            if num_resize > 1
                im_tmp = imresize(im, 1/num_resize);
                net.eval({'x0', im_tmp})
                clear im_tmp;
            else
                net.eval({'x0', im})
            end
            
            
            if ~isempty(p.gpus)
                wait(p.gpu_device);
            end
            
            entire_feature = net.vars(end).value;
            net.reset();
            if ~isempty(p.gpus)
                corr_feature = gather(vl_nnconv(entire_feature, gpuArray(single(long_memory.semantic)),[]));
            else
                corr_feature = vl_nnconv(entire_feature, single(long_memory.semantic),[]);
            end
            clear entire_feature;
            
            
            if ~isempty(p.gpus)
                wait(p.gpu_device);
            end
            
            ratio_map = bsxfun(@rdivide, corr_feature,reshape(best_score,[1, 1, length(best_score)]));
            score_map = ratio_map > p.score_threshold;
            candidates_map = ratio_map.*score_map;
            
            % coarse candidate
            cand_count = 0;
            cand_col = []; cand_row = [];
            %% extract coarse candidate positions
            while 1
                if sum(candidates_map) == 0
                    if cand_count == 0
                        if max(ratio_map(:)) > p.score_threshold2
                            cand_count = cand_count + 1;
                            max_v = max(ratio_map(:));
                            [max_r, max_c, ~] = ind2sub(size(ratio_map), find(ratio_map==max_v));
                            max_r = max_r(1); max_c = max_c(1);
                            cand_row = cat(1, cand_row, max_r);
                            cand_col = cat(1, cand_col, max_c);
                            no_cand = 0;
                        else
                            cand_count = cand_count + 1;
                            max_v = max(ratio_map(:));
                            [max_r, max_c, ~] = ind2sub(size(ratio_map), find(ratio_map==max_v));
                            max_r = max_r(1); max_c = max_c(1);
                            cand_row = cat(1, cand_row, max_r);
                            cand_col = cat(1, cand_col, max_c);
                            no_cand = 1;
                        end
                    end
                    break;
                else
                    if cand_count == p.num_candidate
                        break;
                    end
                    no_cand=0;
                    cand_count = cand_count + 1;
                    max_v = max(candidates_map(:));
                    [max_r, max_c, ~] = ind2sub(size(candidates_map), find(candidates_map==max_v));
                    max_r = max_r(1); max_c = max_c(1);
                    cand_row = cat(1, cand_row, max_r);
                    cand_col = cat(1, cand_col, max_c);
                    candidates_map(max(max_r-p.cand_region,1):min(max_r+p.cand_region, size(candidates_map,1)), max(max_c-p.cand_region,1):min(max_c+p.cand_region,size(candidates_map,2)),:) = 0;
                end
            end
            %% Searching coarse location
            if cand_count ~= 0
                pow_pool = pow2(p.num_pool) * num_resize;
                if p.center_refine_flag
                    cand_row = round(pow_pool*(cand_row + size(long_memory.semantic,1)/2) + pow_pool/2);
                    cand_col = round(pow_pool*(cand_col + size(long_memory.semantic,2)/2) + pow_pool/2);
                else
                    cand_row = pow_pool*(cand_row + floor(size(long_memory.semantic,1)/2) );
                    cand_col = pow_pool*(cand_col + floor(size(long_memory.semantic,2)/2) );
                end
                cand_peak_best = -inf;
                
                
                redetect_scaledInstance = first_s_x .* redetect_scales;
                scaledTarget = [first_targetSize(1) .* redetect_scales; first_targetSize(2) .* redetect_scales];
                % extract scaled crops for search region x at previous target position
                
                % first -> long features
                if ~isempty(p.gpus)
                    redetect_z_features= repmat(gpuArray(single(z_features)), [1 1 1 redetect_num_scales]);
                else
                    redetect_z_features= repmat(single(z_features), [1 1 1 redetect_num_scales]);
                end
                for j = 1 : cand_count
                    cand_position = [cand_row(j) cand_col(j)];
                    clear x_crops;
                    x_crops = make_scale_pyramid(im, cand_position, redetect_scaledInstance, p.instanceSize, avgChans, stats, p);
                    % evaluate the offline-trained network for exemplar x features
                    
                    if p.redetection_p_flag
                        tmp_scalePenalty = p.scalePenalty; tmp_wInfluence = p.wInfluence;
                        p.wInfluence = 0;
                        [newTargetPosition, newScale_tmp, cand_peak] = tracker_eval(net_x, round(s_x), scoreId, redetect_z_features, x_crops, cand_position, window, p);
                        p.scalePenalty = tmp_scalePenalty; p.wInfluence = tmp_wInfluence;
                    else
                        [newTargetPosition, newScale_tmp, cand_peak] = tracker_eval(net_x, round(s_x), scoreId, redetect_z_features, x_crops, cand_position, window, p);
                    end
                    
                    % based on siamese score
                    if cand_peak > cand_peak_best
                        newScale = newScale_tmp;
                        cand_peak_best = cand_peak;
                        targetPosition = gather(newTargetPosition);
                    end
                end
                
                %% Retrieval
                s_x = max(min_s_x, min(max_s_x, (1-p.scaleLR)*s_x + p.scaleLR*redetect_scaledInstance(newScale)));
                s_z = s_x*s_ratio;
                % make semantic input
                
                % retrieval score 
                [semantic_crop, ~] = get_subwindow_tracking(im, targetPosition, [round(first_s_z) round(first_s_z)], [round(s_z) round(s_z)], avgChans, p);
                
                if ~isempty(p.gpus)
                    wait(p.gpu_device);
                end
                if num_resize > 1
                    net.eval({'x0', imresize(semantic_crop, 1/num_resize)})
                else
                    net.eval({'x0', semantic_crop})
                end
                clear semantic_crop;
                clear semantic_feature;
                clear semantic_vector;
                clear semantic_RMAC;
                semantic_feature = net.vars(end).value;
                net.reset();
                semantic_RMAC = RMAC(semantic_feature);
                semantic_vector = max(max(semantic_feature,[],1),[],2);
                semantic_vector = semantic_vector./(sqrt(sum(semantic_vector.^2,3)));
                RMAC_score = mean(sum(bsxfun(@times, squeeze(semantic_RMAC), long_memory.RMAC),1));
                long_score = RMAC_score;
                
                
                if neg_compare
                    targetSize_refine = (1-p.scaleLR)*targetSize + p.scaleLR*[scaledTarget(1,newScale) scaledTarget(2,newScale)];
                    scaledInstance_refine = s_x .* scales;
                    scaledTarget_refine = [targetSize_refine(1) .* scales; targetSize_refine(2) .* scales];
                    
                    if p.compare_crop_flag
                        im_pos = im_change(im, targetPosition, targetSize_refine, 'negative', uint8(avgChans), p.compare_crop_ratio, 1, p);
                        x_crops_pos = make_scale_pyramid(im_pos, targetPosition, scaledInstance_refine, p.instanceSize, avgChans, stats, p);
                    else
                        x_crops_pos = make_scale_pyramid(im, targetPosition, scaledInstance_refine, p.instanceSize, avgChans, stats, p);
                    end
                    
                    if p.redetection_p_flag
                        tmp_scalePenalty = p.scalePenalty; tmp_wInfluence = p.wInfluence;
                        p.scalePenalty = 1; p.wInfluence = 0;
                        [targetPosition_refine, newScale_refine, pos_bestPeak] = tracker_eval(net_x, round(s_x), scoreId, z_features_4d, x_crops_pos, targetPosition, window, p);
                        p.scalePenalty = tmp_scalePenalty; p.wInfluence = tmp_wInfluence;
                    else
                        [targetPosition_refine, newScale_refine, pos_bestPeak] = tracker_eval(net_x, round(s_x), scoreId, z_features_4d, x_crops_pos, targetPosition, window, p);
                    end
                    
                    if p.redetection_refine, targetPosition = gather(targetPosition_refine); end
                    all_neg_pos = [];
                    all_neg_bestPeak = [];
                    for n = 1 : numel(neg_z_features)
                        
                        if p.redetection_p_flag
                            tmp_scalePenalty = p.scalePenalty; tmp_wInfluence = p.wInfluence;
                            p.scalePenalty = 1; p.wInfluence = 0;
                            [neg_pos, ~, new_bestPeak] = tracker_eval(net_x, round(s_x), scoreId, neg_z_features{n}.z_features_neg, x_crops_pos, targetPosition, window, p);
                            p.scalePenalty = tmp_scalePenalty; p.wInfluence = tmp_wInfluence;
                        else
                            [neg_pos, ~, new_bestPeak] = tracker_eval(net_x, round(s_x), scoreId, neg_z_features{n}.z_features_neg, x_crops_pos, targetPosition, window, p);
                        end
                        neg_pos = gather(neg_pos);
                        all_neg_pos = cat(1, all_neg_pos, neg_pos);
                        all_neg_bestPeak = cat(1, all_neg_bestPeak, gather(new_bestPeak));
                    end
                    
                    if p.neg_in_box_flag
                        max_pos = targetPosition + (targetSize_refine - 1) / 2;
                        min_pos = targetPosition - (targetSize_refine - 1) / 2;
                        max_pos = repmat(max_pos, [size(all_neg_pos, 1), 1]);
                        min_pos = repmat(min_pos, [size(all_neg_pos, 1), 1]);
                        delete_idx = find(sum((all_neg_pos <= max_pos) & (all_neg_pos >= min_pos), 2) ~= 2);
                        all_neg_pos(delete_idx, :) = [];
                        all_neg_bestPeak(delete_idx, :) = [];
                    end
                    
                    pos_bestPeak = gather(pos_bestPeak);
                    if ~isempty(all_neg_pos)
                        if sum(p.peak_ratio_redetection * pos_bestPeak < all_neg_bestPeak) > 0
                            no_cand = 1;
                        end
                    else
                        if p.fail_flag
                            short_fail = p.fail_th_tracking - 1;
                        else
                            short_fail = 0;
                        end
                    end
                end
                
                if no_cand == 1
                    oTargetPosition = targetPosition;
                    if p.redetection_refine
                        oTargetSize = (1-p.scaleLR)*targetSize_refine + p.scaleLR*[scaledTarget_refine(1,newScale_refine) scaledTarget_refine(2,newScale_refine)];
                    else
                        oTargetSize = (1-p.scaleLR)*targetSize + p.scaleLR*[scaledTarget(1,newScale) scaledTarget(2,newScale)];
                    end
                    
                    targetPosition = [nan nan];
                    fail_peak = cand_peak_best;
                    
                elseif cand_peak_best/bestPeakFirst > p.peak_ratio_threshold && long_score > p.retrieval_threshold
                    bestPeak = cand_peak_best;
                    long_param.confidence = [];
                    long_param.confidence = bestPeak;
                    p.siamese_score_mean = bestPeak;
                    
                elseif long_score > p.retrieval_high_threshold
                    bestPeak = cand_peak_best;
                    long_param.confidence = [];
                    long_param.confidence = bestPeak;
                    p.siamese_score_mean = bestPeak;
                    
                else
                    oTargetPosition = targetPosition;
                    if p.redetection_refine
                        oTargetSize = (1-p.scaleLR)*targetSize_refine + p.scaleLR*[scaledTarget_refine(1,newScale_refine) scaledTarget_refine(2,newScale_refine)];
                    else
                        oTargetSize = (1-p.scaleLR)*targetSize + p.scaleLR*[scaledTarget(1,newScale) scaledTarget(2,newScale)];
                    end
                    
                    targetPosition = [nan nan];
                    fail_peak = cand_peak_best;
                end
            else
                oTargetPosition = targetPosition;
                targetPosition = [nan nan];
            end
        end
        long_memory.time = long_memory.time + 1;
        
        
        %% scale allocation
        if redetection == 0
            s_x = max(min_s_x, min(max_s_x, (1-p.scaleLR)*s_x + p.scaleLR*scaledInstance(newScale)));
            targetSize = (1-p.scaleLR)*targetSize + p.scaleLR*[scaledTarget(1,newScale) scaledTarget(2,newScale)];
            
        else
            if (isnan(targetPosition(1))) == 0
                s_x = max(min_s_x, min(max_s_x, (1-p.scaleLR)*s_x + p.scaleLR*redetect_scaledInstance(newScale)));
                targetSize = (1-p.scaleLR)*targetSize + p.scaleLR*[scaledTarget(1,newScale) scaledTarget(2,newScale)];
            end
        end
        
        
        if (isnan(targetPosition(1))) == 0
            oTargetPosition = targetPosition; % .* frameSize ./ newFrameSize;
            oTargetSize = targetSize; % .* frameSize ./ newFrameSize;
        else
            
        end
        
        if (isnan(targetPosition(1))) == 0
            confidence = gather(long_score);
        else
            confidence = gather(long_score * (cand_peak_best/bestPeakFirst));
        end
    else
        % at the first frame output position and size passed as input (ground truth)
    end
    i = i + 1;
    
    rectPosition = [oTargetPosition([2,1]) - oTargetSize([2,1])/2, oTargetSize([2,1])];
    
    if strcmp(p.type, 'vot')
        handle = handle.report(handle, rectPosition, confidence);
    end
    runtime = toc;
    
    
    
    if p.visualization
        if isempty(videoPlayer)
            drawnow
            fprintf('Frame %d\n', p.startFrame+i);
        else
            im = gather(im)/255;
            im2 = im;
            if sum(isnan(rectPosition))~=0
                rectPosition = [1, 1, 1, 1];
            end
            
            im = insertShape(im, 'Rectangle', rectPosition, 'LineWidth', 2, 'Color', 'yellow');
            if redetection == 1
                if ~isempty(cand_col)
                    im = insertMarker(im, [cand_col(:), cand_row(:)],'x','size', 5);
                end
                
            end
            
            if ~isempty(all_neg_pos)
                im = insertMarker(im, [all_neg_pos(:, 2), all_neg_pos(:, 1)],'x','size', 5, 'Color', 'red');
            end
            im = insertText(im, [1 1], [num2str(gather(bestPeak)) '/' num2str(gather(bestPeakFirst)), ', ratio : ' num2str(gather(bestPeak/bestPeakFirst))],'FontSize',20);
            im = insertText(im, [size(im,1) 1], ['mean : ' num2str(gather(p.siamese_score_mean))],'FontSize',20);
            im = insertText(im, [1 100], [num2str(gather(fail_peak)) '/' num2str(bestPeakFirst)],'FontSize',20);
            im = insertText(im, [1 30], ['short-term length : ' num2str(size(short_memory.feature,4)) ', long-term length: ', num2str(gather(long_memory.memory_length))],'FontSize',20);
            im = insertText(im, [1 60], ['long-term continue : ' num2str(gather(long_sequence)) ', R-MAC : ', num2str(gather(long_score))],'FontSize',20);
            im = insertText(im, [1 size(im,1)-30], ['time : ' num2str(gather(runtime)) ', confidence : ', num2str(gather(confidence))],'FontSize',20);
            if p.compare_result == 1
                im = insertShape(im, 'Rectangle', my_result(i,:), 'LineWidth', 2, 'Color', 'red');
                im = insertText(im, [size(im,1) size(im,1)-30], ['confidence : ', num2str(gather(result_confidence(i)))],'FontSize',20);
                
            end
            step(videoPlayer, im);
            
        end
    end
    
    if p.store == 1
        f_struct = write_txt(f_struct, [], [], [],rectPosition, confidence, runtime, 2); % write (while-loop)
    end
    
end

if p.store == 1
    write_txt(f_struct, [], [], [], [], [], [], 3); % close (final)
end
if strcmp(p.type, 'vot')
    keepvars = {'handle', 'p'};
else
    keepvars = {'p'};
end
clearvars('-except', keepvars{:});
if strcmp(p.type, 'vot')
    handle.quit(handle);
end

end

