close all; clc; clear all;
% base_path =  'D:\tracking\Dataset\OTB-50'
% base_path = 'D:\tracking\vot-tir2016'
base_path = 'C:\Users\kaist\Desktop\VOT2018\votlt2018\Sequences';

% seq_data = importdata('D:\tracking\vot-tir2016\list.txt')
video_path = 'car8';

[imgFiles, targetPosition, targetSize, img_files] = load_video_info(base_path, video_path);
% im = (single(imgFiles{1}));
im = single(imread(img_files{1}));
videoPlayer = vision.VideoPlayer('Position', [100 100 [size(im,2), size(im,1)]+30]);
my_result = importdata(['C:\Users\kaist\Desktop\VOT2018\votlt2018\results\scale1\longterm\' video_path '/' video_path '_001.txt']);
confidence = importdata(['C:\Users\kaist\Desktop\VOT2018\votlt2018\results\scale1\longterm\' video_path '/' video_path '_001_confidence.value']);;
% dsst_result = importdata([base_path '/' video_path '/dsst_results.txt']);
% ground_truth = importdata([base_path '/' video_path '/groundtruth.txt']);
% tld_truth = i
% if size(ground_truth,2) == 8
%    ground_truth = [min(ground_truth(:,1:2:end),[],2), min(ground_truth(:,2:2:end),[],2),...
%           (max(ground_truth(:,1:2:end),[],2)-min(ground_truth(:,1:2:end),[],2)),...
%           (max(ground_truth(:,2:2:end),[],2)-min(ground_truth(:,2:2:end),[],2))]; 
% end

for frameIdx = 1:numel(img_files)
    if frameIdx == 2
        1
    end
%     im = single(imgFiles{frameIdx});
    im = single(imread(img_files{frameIdx}));
    if isempty(videoPlayer)
%         figure(1), imshow(im/255);
%         figure(1), rectangle('Position', ground_truth(frameIdx,:), 'LineWidth', 4, 'EdgeColor', 'g');
%           figure(1), rectangle('Position', dsst_result(frameIdx,:), 'LineWidth', 4, 'EdgeColor', 'm');
%         figure(1), rectangle('Position', ECO_result(frameIdx,:), 'LineWidth', 4, 'EdgeColor', 'b');
%         figure(1), rectangle('Position', siamese_result(frameIdx,:), 'LineWidth', 4, 'EdgeColor', 'y');
%        drawnow
        fprintf('Frame %d\n', frameIdx);
    else
        im = gather(im)/255;
%         im = insertShape(im, 'Rectangle', ground_truth(frameIdx,:), 'LineWidth', 2, 'Color', 'green');
%         im = insertShape(im, 'Rectangle', dsst_result(frameIdx,:), 'LineWidth', 2, 'Color', 'magenta');
%         im = insertShape(im, 'Rectangle', ECO_result(frameIdx,:), 'LineWidth', 2, 'Color', 'blue');
        im = insertShape(im, 'Rectangle', my_result(frameIdx,:), 'LineWidth', 2, 'Color', 'yellow');
        im = insertText(im, [1 size(im,1)-30], ['confidence : ', num2str(gather(confidence(frameIdx)))],'FontSize',20);

%         legend('ground truth');
        % Display the annotated video frame using the video player object.
        step(videoPlayer, im);
    end
end