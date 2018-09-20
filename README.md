## A Memory Model based on the Siamese Network for Long-term Tracking (MMLT)
- - - -
>This instructions are for Win10
>Pre-requisites : GPU (CPU is also available, but slow), CUDA (we used 8.0), cuDNN (we used 7.1), MATLAB (we used 2017a), MatConvNet (we used 1.0-beta25)
 
1. Setting MatConvNet

* Download the MatConvNet and cuDNN in the "matconvnet" folder (http://www.vlfeat.org/matconvnet/)
* If you already have the MatConvNet, move it to these folders or change the direction at the "setup_paths.m" file.
 
2. Install
* Go to "/runfiles/" and run the m-file "install.m"
 ( The runfile automatically download the pretrained network (2016-08-17.net.mat) into the "pretrained" folder [http://www.robots.ox.ac.uk/~luca/siamese-fc.html] and
    the pretrained network (imagenet-vgg-verydeep-19.mat) into the "pretrained" folder [http://www.vlfeat.org/matconvnet/pretrained/] )
  ( The runfile also complie the matconvnet => GPU : vl_compilenn('enableGpu', true), CPU : vl_compilenn; )
  ( If cudnn and cuda are not available in your PC, it will be not operated. So Please cheack http://www.vlfeat.org/matconvnet/install/ )
 
3. Demo (It is not required for VOT integration, but try it for convenience)
* Go to "/runfiles/" and run the m-file "run_demo.m".
* If you want to do experiments by using other datasets, change the directory or move the dataset into the "sequences" folder

4. VOT Integration
* Go to "/runfiles/" and move the m-file "tracker_MMLT" to your VOT workspace
* Change the root_dir to the directory including the "MMLT" folder
* Run the m-file "run_test.m"
* Run the m-file "run_experiments.m"

- - - -

> If you get an error "gpuarray", check the readme file
> If you get an error "out of memory" on the GPU, increase p.gpu_memory_resize_add in setting_parameters.m file
