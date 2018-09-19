function p = setting_parameters(varargin)


%% Common parameters
% vl_compilenn;
p.type = varargin{1}; % 'vot', 'demo'
p.gpus = varargin{2}; % GPU
if strcmp(p.type, 'vot') % vot version
    p.visualization = 0;
    p.store = 0;
    p.compare_result = 0;
elseif strcmp(p.type, 'demo') % for demo (need datasets)
    p.visualization = 1;
    p.store = 1;
    p.compare_result = 0;
    p.startFrame = 1;
    p.compare_result_name = 'demo';
    p.folder_name = 'demo';
    p.root_name = [varargin{3}, 'results\'];
    p.seq_base_path = [varargin{3}, 'Sequences'];
    p.seq = importdata([varargin{3}, 'Sequences\list.txt']);
end
[pathstr, ~, ~] = fileparts(mfilename('fullpath'));
idx = findstr(pathstr, '\');
pathstr = pathstr(1 : idx(end));
p.net_base_path = [pathstr, 'pretrained\'];

if ~isempty(p.gpus)
    p.gpu_device = gpuDevice(1);
end



%% Parameters for siamese network
p.numScale = 3;
p.scaleStep = 1.0375;
p.scalePenalty = 0.9745;
p.scaleLR = 0.85; % damping factor for scale update
p.wInfluence = 0.176; % windowing influence (in convex sum)
p.responseUp = 16; % upsampling the small 17x17 response helps with the accuracy
p.windowing = 'cosine'; % to penalize large displacements
p.net = '2016-08-17.net.mat';
p.redetection_net = 'imagenet-vgg-verydeep-19.mat';
p.neg_in_box_flag = 1;
p.fout = -1;
p.exemplarSize = 127;  % input z size
p.instanceSize = 255;  % input x size (search region)
p.scoreSize = 17;
p.totalStride = 8;
p.contextAmount = 0.5; % context amount for the exemplar
p.subMean = false;
p.prefix_z = 'a_'; % used to identify the layers of the exemplar
p.prefix_x = 'b_'; % used to identify the layers of the instance
p.prefix_join = 'xcorr';
p.prefix_adj = 'adjust';
p.id_feat_z = 'a_feat';
p.id_score = 'score';


%% Parameters for redetection
p.num_limit_pixel = 1000000; % gpu memory
p.center_refine_flag = 0;
p.num_pool = 4;
p.max_redetection_num = 15;
p.num_candidate = 3; % redetection candidate
p.cand_region = 2.0;
% p.min_scale_after = 0.015;
p.min_scale_after = 0.015;
p.max_scale_after = 1.0;
p.min_scale_before = 0.05;
p.max_scale_before = 4.5;
p.redetection_p_flag = 1;
p.pow_value = 2;
p.pow_scale = [-36, -25, -16, -9, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 9, 16];
p.score_threshold = 0.3;  % redetection threshold
p.score_threshold2 = 0.2;  % redetection threshold



%% Parameters for negative samples
p.neg_siamese_size_flag = 1; % 1 : based on siamese size, 2 : based on GT size
p.pos_ratio = 1.0; % positive ratio (candidate region)
p.neg_ratio = 1.0; % negative ratio (discard the GT bounding box)
p.neg_crop_flag = 1; % 1 : siamese size, 2 : target_size (more smaller)
p.permit_ratio = 0.9; % -1 (restricted) ~ 1 (released)
p.compare_crop_flag = 0; % 1 : w/ crop , 0 : w/o crop
p.compare_crop_ratio = 1.5; % crop ratio
p.peak_ratio_short = 1.0;
p.peak_ratio_redetection = 1.0;
p.long_cnt_restart = 1;
p.redetection_refine = 1;
p.num_neg_siamese = 8;
p.num_neg_cnn = 8;
p.num_max_negative_search = 20;
p.neg_cnn_first_center = 1;


%% Parameters for fail cases
p.fail_flag = 1;
if p.fail_flag
    p.fail_th_tracking = 30;
    p.fail_th_redetection = 5;
else
    p.fail_th_tracking = 50000;
    p.fail_th_redetection = 25000;
end


%% short/long - term parameters
p.siamese_score_num_save = 40; % score save length
p.siamese_score_threshold = 0.6; % BestPeak/BestPeakMean threshold
p.siamese_score_mean = 0;        % BestPeakMean
p.short_memory_damping = 60;     % when short term filter update,  s value of exp(-t/s)
p.short_long_damping = 150;
p.short_memory_frames = 60;   % short-term memory length
p.long_memory_frames = 40;     % long-term memory length
p.long_memory_interval = 10;  % long-term memory save interval
p.forget_h = 120;
p.forget_strength = 1;
p.forgetting_peak = 1;
p.forgetting_threshold = 0.1;
p.forgetting_up = 1.3;
p.retrieval_threshold = 0.6;
p.retrieval_high_threshold = 0.7;
p.short_term_store_threshold = 0.5;
p.filter_short_long_ratio = 0.4;
p.peak_ratio_threshold = 0.35;  % the score threshold of bestPeak/bestPeakFirst when redetection, tracking 

%% GPU memory error
p.gpu_memory_resize_basic = 1; % the basic ratio of resize
p.gpu_memory_resize_add = 0; % the ratio of resize for adding (if GPU memory error occurs, increase this value)


end







