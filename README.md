This is a Tensorflow implementation for the paper 'Fully Automatic Video Colorization with Self-Regularization and Diversity'.


## Quick inference( without refinement network) 
For convenience, we also provide the version without the refinement network.
It's easier to use.

(1) You don't need to generate optical flow by PWC-Net for refinement. 

(2) Less libraries are required.

(3) It could also be used for single image colorization. 

First, download the ckpt. 

```
python download_models.py
unzip ckpt_woflow.zip
```

For video colorization
```
python main_woflow.py --model ckpt_woflow --use_gpu 1 --test_dir /PATH/TO/TEST/DIR

e.g.
python main_woflow.py --model ckpt_woflow --use_gpu 1 --test_dir test_sample0
```

For single image colorization
```
python main_woflow.py --model ckpt_woflow --use_gpu 1 --test_img /PATH/TO/TEST_IMG

e.g.
python main_woflow.py --model ckpt_woflow --use_gpu 1 --test_img test_sample0/frame_000980.jpg 
```

Results are saved in ./ckpt_woflow/ folder.

## Requirement
Required python libraries:

```
tensorflow 1.2.0
OpenCV 3.4.2.16
```

Tested on Ubuntu 16.04 + Nvidia 1080Ti + Cuda 8.0 + cudnn 7.0


## Training
After downloading the dataset,

```
python main.py --model YOUR_MODEL_NAME --data_dir data
```

### Using your own data
Arrange your data in the following format:

```
+data
-----+JPEGImages
----------------+480p
---------------------+VideoFrames1
---------------------+VideoFrames2
-----+FLOWImages
----------------+Forward
-----------------------+VideoFrames1
-----------------------+VideoFrames2
----------------+Backward
-----------------------+VideoFrames1
-----------------------+VideoFrames2
-----+FLOWImages_GRAY
--------------------+Forward
----------------------------+VideoFrames1
----------------------------+VideoFrames2
--------------------+Backward
----------------------------+VideoFrames1
----------------------------+VideoFrames2
```

Please generate the optical flow by PWC-Net. If you use FlowNet2 or other methods, please make sure the file format is the same.

## Citation
If you use our code or paper, please cite:

```
@InProceedings{Lei_2019_CVPR,
author = {Lei, Chenyang and Chen, Qifeng},
title = {Fully Automatic Video Colorization With Self-Regularization and Diversity},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```

## Contact
If you have any question, please feel free to contact me (Chenyang LEI, leichenyang7@gmail.com)

