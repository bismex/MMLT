tracker_label = 'MMLT';
root_dir = 'C:\Users\choi-home\Desktop\MMLT\runfiles'; % The directory should be changed !!!!!
runfile_name = 'run_votlt2018'; % If you want to run the cpu version, go to "runfiles/run_votlt2018.m"
runfile_name = [runfile_name, '(''', root_dir, ''')'];
tracker_command = generate_matlab_command(runfile_name, {root_dir});
tracker_interpreter = 'matlab';


% If you get an error "gpuarray", check the readme file.
% If you get an error "out of memory" on the GPU, increase p.gpu_memory_resize_add in setting_parameters.m file.
