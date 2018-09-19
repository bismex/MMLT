
[home_dir, ~, ~] = fileparts(mfilename('fullpath'));

% donwload network
cd .././pretrained
if ~(exist('2016-08-17.net.mat', 'file') == 2)
    disp('Downloading the network "2016-08-17.net.mat" from "http://www.robots.ox.ac.uk/~luca/stuff/siam-fc_nets/2016-08-17.net.mat"...')
    urlwrite('http://www.robots.ox.ac.uk/~luca/stuff/siam-fc_nets/2016-08-17.net.mat', '2016-08-17.net.mat')
    disp('Done!')
end
if ~(exist('imagenet-vgg-verydeep-19.mat', 'file') == 2)
    disp('Downloading the network "imagenet-vgg-verydeep-19.mat" from "http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat"...')
    urlwrite('http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat', 'imagenet-vgg-verydeep-19.mat')
    disp('Done!')
end
cd(home_dir)

cd .././matconvnet/matlab
% This is CPU version
vl_compilenn;
% This is GPU version
disp('Trying to compile MatConvNet with GPU support')
vl_compilenn('enableGpu', true); 
% If cudnn and cuda are not available in your PC, it will be not operated. So Please cheack http://www.vlfeat.org/matconvnet/install/
cd(home_dir)