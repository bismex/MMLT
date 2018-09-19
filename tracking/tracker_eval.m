% -------------------------------------------------------------------------------------------------------------------------
function [newTargetPosition, bestScale, bestPeak] = tracker_eval(net_x, s_x, scoreId, z_features, x_crops, targetPosition, window, p)
%TRACKER_STEP
%   runs a forward pass of the search-region branch of the pre-trained Fully-Convolutional Siamese,
%   reusing the features of the exemplar z computed at the first frame.
%
%   Luca Bertinetto, Jack Valmadre, Joao F. Henriques, 2016
% -------------------------------------------------------------------------------------------------------------------------
    % forward pass, using the pyramid of scaled crops as a "batch"
    num_scales = size(x_crops,4);
    net_x.eval({p.id_feat_z, z_features, 'instance', x_crops});
%     tmp_mat = gather(net_x.vars(scoreId).value);
    tmp_mat = net_x.vars(scoreId).value;
    net_x.reset();
    responseMaps = reshape(tmp_mat, [p.scoreSize p.scoreSize num_scales]);
    clear tmp_mat;
    
    if ~isempty(p.gpus)
        wait(p.gpu_device);
    end
    if ~isempty(p.gpus)
        responseMapsUP = gpuArray(single(zeros(p.scoreSize*p.responseUp, p.scoreSize*p.responseUp, num_scales)));
    else
        responseMapsUP = single(zeros(p.scoreSize*p.responseUp, p.scoreSize*p.responseUp, num_scales));
    end
%     end
    % Choose the scale whose response map has the highest peak
    if num_scales>1
        currentScaleID = ceil(num_scales/2);
        bestScale = currentScaleID;
        bestPeak = -Inf;
        for s=1:num_scales
            if p.responseUp > 1
                % upsample to improve accuracy
%                 responseMapsUP(:,:,s) = imresize(gather(responseMaps(:,:,s)), p.responseUp, 'bicubic');
                responseMapsUP(:,:,s) = imresize(responseMaps(:,:,s), p.responseUp, 'bicubic');
            else
                responseMapsUP(:,:,s) = responseMaps(:,:,s);
            end
            thisResponse = responseMapsUP(:,:,s);
            % penalize change of scale
            if s~=currentScaleID, thisResponse = thisResponse * p.scalePenalty; end
            thisPeak = max(thisResponse(:));
            if thisPeak > bestPeak, bestPeak = thisPeak; bestScale = s; end
        end
        responseMap = responseMapsUP(:,:,bestScale);
    else
        responseMap = responseMapsUP;
        bestScale = 1;
    end
%     PSR = (max(responseMap(:)) - mean(responseMap(:)))/(std(responseMap(:)));
%     bestPeak = bestPeak * PSR;
%     figure(15), imagesc(responseMap);
    % make the response map sum to 1
    responseMap = responseMap - min(responseMap(:));
    responseMap = responseMap / sum(responseMap(:));
    % apply windowing
    responseMap = (1-p.wInfluence)*responseMap + p.wInfluence*window;
    [r_max, c_max] = find(responseMap == max(responseMap(:)), 1);
    [r_max, c_max] = avoid_empty_position(r_max, c_max, p);
    p_corr = [r_max, c_max];
    % Convert to crop-relative coordinates to frame coordinates
    % displacement from the center in instance final representation ...
    disp_instanceFinal = p_corr - ceil(p.scoreSize*p.responseUp/2);
    % ... in instance input ...
    disp_instanceInput = disp_instanceFinal * p.totalStride / p.responseUp;
    % ... in instance original crop (in frame coordinates)
    disp_instanceFrame = disp_instanceInput * s_x / p.instanceSize;
    % position within frame in frame coordinates
    newTargetPosition = targetPosition + disp_instanceFrame;
    clear tmp_mat;
    clear responseMapsUP;
    clear responseMaps;
    clear responseMap;
    clear thisResponse;
end

function [r_max, c_max] = avoid_empty_position(r_max, c_max, params)
    if isempty(r_max)
        r_max = ceil(params.scoreSize/2);
    end
    if isempty(c_max)
        c_max = ceil(params.scoreSize/2);
    end
end
