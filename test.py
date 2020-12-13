#tensorflow 1.2.0 is needed
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import utils as utils
import subprocess
import network as net
import loss as loss
import argparse
from glob import glob
import imageio
import json
import math
import random
from PIL import Image

seed = 2020
np.random.seed(seed)
tf.set_random_seed(seed)
random.seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='model_imagecolor', type=str, help="Model Name")
parser.add_argument("--img_path", default='demo_imgs/ILSVRC2012_val_00040251.JPEG', type=str, help="Test dir path")
parser.add_argument("--video_path", default=None, type=str, help="Test dir path")
parser.add_argument('--network', type=str, default='half_hyper_unet',
                            help='chooses which model to use. unet, fcn',
                            choices=["half_hyper_unet", "hyper_unet"])


ARGS = parser.parse_args()
print(ARGS)
os.makedirs("./test_result/{}".format(ARGS.model), exist_ok=True)


os.environ["CUDA_VISIBLE_DEVICES"]=str(np.argmax( [int(x.split()[2]) for x in subprocess.Popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]))
config=tf.ConfigProto()
config.gpu_options.allow_growth=True
sess=tf.Session(config=config)


gray1=tf.placeholder(tf.float32,shape=[None,None,None, 1])
rgb1=tf.placeholder(tf.float32,shape=[None,None,None, 3])


def get_model(gray1):
    if ARGS.network == "half_hyper_unet":      
        pred1 = net.dowmsample_unet(gray1, reuse=False)
    elif ARGS.network == "hyper_unet":      
        pred1 = net.VCN(utils.build(tf.tile(gray1, [1,1,1,3])), reuse=False, div_num=1)
    return pred1

with tf.variable_scope(tf.get_variable_scope()):
    with tf.variable_scope('siamese_nework'):
        pred1 = get_model(gray1)

saver=tf.train.Saver(max_to_keep=1000)
sess.run([tf.global_variables_initializer()])
var_restore = [v for v in tf.trainable_variables()]
saver_restore=tf.train.Saver(var_restore)
ckpt=tf.train.get_checkpoint_state('./pretrained_models/' + ARGS.model)
print("contain checkpoint: ", ckpt)
if ckpt:
    print('loaded '+ ckpt.model_checkpoint_path)
    saver_restore.restore(sess, ckpt.model_checkpoint_path)
else:
    print("there is no checkpoint: {}".format('./pretrained_models/' + ARGS.model))

def test():
    if ARGS.video_path is None:
        val_names = [ARGS.img_path]
    else:
        val_names = sorted(glob(ARGS.video_path + '/*'))
    cnt = 0
    print(len(val_names))
    for id in range(len(val_names)):
        gray_image=np.array(Image.open(val_names[id]).convert("L")) / 255.
        basename = val_names[id].split("/")[-1]
        h=gray_image.shape[0] // 32 * 32
        w=gray_image.shape[1] // 32 * 32       
        pred_image = sess.run(pred1, feed_dict={gray1:gray_image[np.newaxis,:h,:w,np.newaxis]})
        if ARGS.video_path is None:
            imageio.imwrite("./test_result/{}/{}".format(ARGS.model, basename.replace(".JPEG", "_gray.png")), gray_image[:h,:w])
            imageio.imwrite("./test_result/{}/{}".format(ARGS.model, basename.replace(".JPEG", "_colorized.png")), pred_image[0])
        else:
            base_path = ARGS.video_path[:-1] if ARGS.video_path.endswith("/") else ARGS.video_path
            os.makedirs(base_path + "_colorized", exist_ok=True)
            imageio.imwrite("{}_colorized/{}".format(base_path, basename), gray_image[:h,:w])
            imageio.imwrite("{}_colorized/{}".format(base_path, basename), pred_image[0])

test()