This is a Tensorflow implementation for the paper 'Fully Automatic Video Colorization with Self-Regularization and Diversity'.

## To do

```
- Upload training data, testing data and ckpt of the whole model.
- Upload the training code, inference code of the whole model 
```


## Quick inference 
For convenience, we also provide the version without the refinement network.
It's easier to use.

(1) You don't need to generate optical flow by PWC-Net for refinement. 

(2) Less libraries are required.

(3) It could also be used for single image colorization. 

First, download the ckpt. 

```
pip install gdown
gdown https://drive.google.com/uc?id=1yL8x7RL_82Mvyh_ebmh3uF6wiPOO2dU7
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
