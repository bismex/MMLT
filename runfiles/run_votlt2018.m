function run_votlt2018(varargin)

gpu = 1; % GPU : 1, CPU : []
setup_paths(varargin{1}, gpu); 
p = setting_parameters('vot', gpu);
tracker_VOT(p);