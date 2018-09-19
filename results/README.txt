<How to run> MMLT

This instructions are for Win10.
Pre-requisites : GPU (CPU is also available, but slow), CUDA (we used 8.0), cuDNN (we used 7.1), MATLAB (we used 2017a), MatConvNet (we used 1.0-beta25)

1. Setting MatConvNet
  1.1 Download the MatConvNet and cuDNN in the "matconvnet" folder (http://www.vlfeat.org/matconvnet/)
  1.2 If you already have the MatConvNet, move it to these folders or change the direction at the "setup_paths.m" file.
2. Install
  2.1 Go to "/runfiles/" and run the m-file "install.m"
  ( The runfile automatically download the pretrained network (2016-08-17.net.mat) into the "pretrained" folder [http://www.robots.ox.ac.uk/~luca/siamese-fc.html] and
    the pretrained network (imagenet-vgg-verydeep-19.mat) into the "pretrained" folder [http://www.vlfeat.org/matconvnet/pretrained/] )
  ( The runfile also complie the matconvnet => GPU : vl_compilenn('enableGpu', true), CPU : vl_compilenn; )
  ( If cudnn and cuda are not available in your PC, it will be not operated. So Please cheack http://www.vlfeat.org/matconvnet/install/ )
3. Demo (It is not required for VOT integration, but try it for convenience)
  3.1 Go to "/runfiles/" and run the m-file "run_demo.m".
  3.2 If you want to do experiments by using other datasets, change the directory or move the dataset into the "sequences" folder
4. VOT Integration
  4.1 Go to "/runfiles/" and move the m-file "tracker_MMLT" to your VOT workspace
  4.2 Change the root_dir to the directory including the "MMLT" folder
  4.3 Run the m-file "run_test.m"
  4.4 Run the m-file "run_experiments.m"


If you get an error "gpuarray", check the readme file.
If you get an error "out of memory" on the GPU, increase p.gpu_memory_resize_add in setting_parameters.m file.

<Code reference>
@inproceedings{bertinetto2016fully,
  title={Fully-Convolutional Siamese Networks for Object Tracking},
  author={Bertinetto, Luca and Valmadre, Jack and Henriques, Jo{\~a}o F and Vedaldi, Andrea and Torr, Philip H S},
  booktitle={ECCV 2016 Workshops},
  pages={850--865},
  year={2016}
}


