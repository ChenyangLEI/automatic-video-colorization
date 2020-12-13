
This is a Tensorflow implementation for the CVPR 2019 paper 'Fully Automatic Video Colorization with Self-Regularization and Diversity'.


![alt text](https://github.com/ChenyangLEI/Fully-Automatic-Video-Colorization-with-Self-Regularization-and-Diversity/blob/master/Teaser.PNG)

More results are shown on our project website https://leichenyang.weebly.com/project-color.html

## News
We propose a novel and general framework [Deep-Video-Prior](https://chenyanglei.github.io/DVP/index.html) 
that can address the temporal inconsistency problem given an input video and a processed video.
We can obtain high-quality video using a single-image colorization method and our novel framework.
 


## Quick inference( without refinement network) 
```
conda env create -f environment.yml
conda activate automatic-video-colorization
bash pretrained_models/download_models.sh
python test.py --task model_latest
```

## Dependency
This code is based on tensorflow. It has been tested on Ubuntu 18.04 LTS.

Anaconda is recommended: Ubuntu 18.04 | Ubuntu 16.04

After installing Anaconda, you can setup the environment simply by

```
conda env create -f environment.yml
conda activate automatic-video-colorization
```

## Usage
### Image colorization

### Video colorization without optical flow
First, download the ckpt. 

```
python download_models.py
unzip ckpt_woflow.zip
```

For video colorization, the video should be split to frames first, i.e., transfer video format (.mp4/.avi) to image format (.jpg/.png)
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

```
python main.py --model YOUR_MODEL_NAME --data_dir data
```

### Prepare the dataset
For the video dataset, please download the DAVIS dataset and generate the optical flow by PWC-Net by yourself. If you want to use FlowNet2 or other methods, please make sure the file format is the same.

At last, please arrange your data in the following format:

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

For the image dataset, please download the ImageNet dataset.

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

