function setup_paths(varargin)

if isempty(varargin{1})
    [pathstr, ~, ~] = fileparts(mfilename('fullpath'));
else
    pathstr = varargin{1};
end
idx = findstr(pathstr, '\');
pathstr = pathstr(1 : idx(end));

addpath([pathstr, 'pretrained/']);
addpath([pathstr, 'tracking/']);
addpath([pathstr, 'util/']);
addpath([pathstr, 'matconvnet/matlab/mex/']);
addpath([pathstr, 'matconvnet/matlab/']);
addpath([pathstr, 'matconvnet/matlab/simplenn/']);
    
vl_setupnn;
