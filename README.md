
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
python test.py 
```
the results are saved in test_result/

## Dependency
### Environment
This code is based on tensorflow. It has been tested on Ubuntu 18.04 LTS.

Anaconda is recommended: [Ubuntu 18.04](https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-18-04)
| [Ubuntu 16.04](https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-16-04)

After installing Anaconda, you can setup the environment simply by

```
conda env create -f environment.yml
conda activate automatic-video-colorization
```

### Pretrained-models and VGG-Models
```
bash download_pretrained.sh
```


## Usage
### Image colorization
We provide the ckpt to colorize a single image, the temporal consistency is not as good as video colorization but the colorization performance is better.

You can colorization a single image, e.g.:
```
python test.py --img_path PATH/TO/IMAGE
e.g.,
python test.py --img_path demo_imgs/ILSVRC2012_val_00040251.JPEG
```

or colorize the images in a folder by:
```
python test.py --img_path PATH/TO/FOLDER
e.g., 
python test.py --img_path demo_imgs/
```
The results are saved in test_result/

### Video colorization without optical flow
Step1. the video should be split to frames first:
```
python video_utils.py --video2frames  --video_dir demo.mkv --out_frames_dir demo_framesdir
```

Step2. Colorizing blanc-and-white frames (you can also use the image colorization pretrained model to colorize the frames):
```
python test_div_video.py --use_gpu 1 --video_path demo_framesdir 
```
Results are saved in video_path_colorized, e.g., demo_framesdir_colorized

Step3. Convert colorized frames to video, note that you need to check the fps of original video
```
python video_utils.py --frames2video --colorized_video_dir demo_colorized.mp4 --colorized_frames_dir demo_framesdir_colorized --fps 24
```

Step4. Add the sound of original video (again, you need to make sure the fps of colorized vide is consistent with the original video)
```
python video_utils.py --add_sound --colorized_video_dir demo_colorized.mp4 --video_dir demo.mkv
```



## Training

```
python main_whole.py --model YOUR_MODEL_NAME --data_dir data
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

