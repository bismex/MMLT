close all;
clear; clc;
fclose all;

gpu = 1; % 1 : GPU / [] : CPU
setup_paths([], gpu);
[pathstr, ~, ~] = fileparts(mfilename('fullpath'));
idx = findstr(pathstr, '\'); demo_directory = pathstr(1 : idx(end));
p = setting_parameters('demo', gpu, demo_directory);
for s=1:numel(p.seq)
    close all;
    p.video = p.seq{s};
    tracker_VOT(p);
end

% If you get an error "gpuarray", check the readme file.
% If you get an error "out of memory" on the GPU, increase p.gpu_memory_resize_add in setting_parameters.m file.


