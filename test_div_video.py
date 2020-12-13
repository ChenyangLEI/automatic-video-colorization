#tensorflow 1.2.0 is needed
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os,time,scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import utils as utils
import imageio 
import subprocess
import argparse
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='model_div_woflow', type=str, help="Model Name")
parser.add_argument("--use_gpu", default=1, type=int, help="Use gpu or not")
parser.add_argument("--video_path", default='demo_vid', type=str, help="Test video dir")
parser.add_argument("--img_path", default=None, type=str, help="Test image path")
parser.add_argument("--output_dir", default=None, type=str, help="Output frames dir")


ARGS = parser.parse_args()
video_path = ARGS.video_path
img_path = ARGS.img_path
model=ARGS.model
print(ARGS)

if not ARGS.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"]=''  
else:
    os.environ["CUDA_VISIBLE_DEVICES"]=str(np.argmax( [int(x.split()[2]) for x in subprocess.Popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]))

def identity_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        array = np.zeros(shape, dtype=float)
        cx, cy = shape[0]//2, shape[1]//2
        for i in range(np.minimum(shape[2],shape[3])):
            array[cx, cy, i, i] = 1
        return tf.constant(array, dtype=dtype)
    return _initializer

def lrelu(x):
    return tf.maximum(x*0.2,x)

def bilinear_up_and_concat(x1, x2, output_channels, in_channels, scope):
    with tf.variable_scope(scope):
        upconv = tf.image.resize_images(x1, [tf.shape(x1)[1]*2, tf.shape(x1)[2]*2] )
        upconv.set_shape([None, None, None, in_channels])
        upconv = slim.conv2d(upconv,output_channels,[3,3], rate=1, activation_fn=None, weights_initializer=tf.contrib.layers.xavier_initializer(),scope='up_conv1')
        upconv_output =  tf.concat([upconv, x2], axis=3)
        upconv_output.set_shape([None, None, None, output_channels*2])
    return upconv_output

def VCN(input, channel=32, output_channel=3,reuse=False,ext=""):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    conv1=slim.conv2d(input,channel,[1,1], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv1_1')
    conv1=slim.conv2d(conv1,channel,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv1_2')
    pool1=slim.max_pool2d(conv1, [2, 2], padding='SAME' )
    conv2=slim.conv2d(pool1,channel*2,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv2_1')
    conv2=slim.conv2d(conv2,channel*2,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv2_2')
    pool2=slim.max_pool2d(conv2, [2, 2], padding='SAME' )
    conv3=slim.conv2d(pool2,channel*4,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv3_1')
    conv3=slim.conv2d(conv3,channel*4,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv3_2')
    pool3=slim.max_pool2d(conv3, [2, 2], padding='SAME' )
    conv4=slim.conv2d(pool3,channel*8,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv4_1')
    conv4=slim.conv2d(conv4,channel*8,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv4_2')
    pool4=slim.max_pool2d(conv4, [2, 2], padding='SAME' )
    conv5=slim.conv2d(pool4,channel*16,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv5_1')
    conv5=slim.conv2d(conv5,channel*16,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv5_2')
    up6 =  bilinear_up_and_concat( conv5, conv4, channel*8, channel*16, scope=ext+"g_up_1" )
    conv6=slim.conv2d(up6,  channel*8,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv6_1')
    conv6=slim.conv2d(conv6,channel*8,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv6_2')
    up7 =  bilinear_up_and_concat( conv6, conv3, channel*4, channel*8, scope=ext+"g_up_2" )
    conv7=slim.conv2d(up7,  channel*4,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv7_1')
    conv7=slim.conv2d(conv7,channel*4,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv7_2')
    up8 =  bilinear_up_and_concat( conv7, conv2, channel*2, channel*4, scope=ext+"g_up_3" )
    conv8=slim.conv2d(up8,  channel*2,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv8_1')
    conv8=slim.conv2d(conv8,channel*2,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv8_2')
    up9 =  bilinear_up_and_concat( conv8, conv1, channel, channel*2, scope=ext+"g_up_4" )
    conv9=slim.conv2d(up9,  channel,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv9_1')
    conv9=slim.conv2d(conv9,output_channel*4,[3,3], rate=1, activation_fn=None,  weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv9_2')
    return conv9


config=tf.ConfigProto()
config.gpu_options.allow_growth=True
sess=tf.Session(config=config)
input_i=tf.placeholder(tf.float32,shape=[1,None,None,2])
input_target=tf.placeholder(tf.float32,shape=[1,None,None,6])

with tf.variable_scope(tf.get_variable_scope()):
    with tf.variable_scope('individual'):
        g0=VCN(utils.build(tf.tile(input_i[:,:,:,0:1],[1,1,1,3])), reuse=False)
        g1=VCN(utils.build(tf.tile(input_i[:,:,:,1:2],[1,1,1,3])), reuse=True)

saver=tf.train.Saver(max_to_keep=1000)
sess.run([tf.global_variables_initializer()])

var_restore = [v for v in tf.trainable_variables()]
saver_restore=tf.train.Saver(var_restore)
ckpt=tf.train.get_checkpoint_state("pretrained_models/" + model)
print('loaded '+ ckpt.model_checkpoint_path)
saver_restore.restore(sess,ckpt.model_checkpoint_path)

folder = video_path.split('/')[-1]
output_dir = ARGS.video_path[:-1] + "_colorized" if ARGS.video_path.endswith("/") else ARGS.video_path + "_colorized" 

# if ARGS.output_dir is None:
#     output_dir = "test_result/{}/{}".format(model, folder)
# else:
#     output_dir = ARGS.output_dir
if img_path is None:
    img_names = utils.get_names(video_path)
    ind=0
    for frame_id, img_name in enumerate(img_names):
        im = np.array(Image.open(img_name).convert("L")) / 255.
        h = im.shape[0]//32*32
        w = im.shape[1]//32*32
        im = im[np.newaxis,:h,:w,np.newaxis]
        st=time.time()
        output=sess.run(g0,feed_dict={input_i:np.concatenate((im,im),axis=3)})
        if frame_id % 10 == 0:
            print("test time for colorizing frame %s --> %.3f"%(ind, time.time()-st))
        for idx in range(5):
            os.makedirs("{}/result{}".format(output_dir, idx), exist_ok=True)
        out_all = np.concatenate([output[:,:,:,3*i:3*i+3] for i in range(4)],axis=2)
        for idx in range(4):
            imageio.imwrite("{}/result{}/{:05d}.jpg".format(output_dir, idx+1, ind), 
                np.uint8(np.maximum(np.minimum(output[0,:,:,3*idx:3*idx+3] * 255.0,255.0),0.0)))
        imageio.imwrite("{}/result0/{:05d}.jpg".format(output_dir, ind),np.uint8(np.maximum(np.minimum(out_all[0,:,:,:] * 255.0,255.0),0.0)))    
        imageio.imwrite("{}/result0/input_{:05d}.jpg".format(output_dir, ind),np.uint8(np.maximum(np.minimum(im[0,:,:,0] * 255.0,255.0),0.0)))    
        
        ind+=1

else:
    im = np.array(Image.open(img_name).convert("L")) / 255.
    h=im.shape[0]//32*32
    w=im.shape[1]//32*32
    im=im[np.newaxis,:h,:w,np.newaxis]
    st=time.time()
    output=sess.run(g0,feed_dict={input_i:np.concatenate((im,im),axis=3)})
    print("test time for frame %s --> %.3f"%(img_path, time.time()-st))
    folder = video_path.split('/')[-1]
    out_all = np.concatenate([output[:,:,:,3*i:3*i+3] for i in range(4)],axis=2)
    for idx in range(4):
        imageio.imwrite("test_result/%s/%s_result%d.jpg"%(model,img_path.split('/')[-1][:-4],idx+1),np.uint8(np.maximum(np.minimum(output[0,:,:,3*idx:3*idx+3] * 255.0,255.0),0.0)))
    imageio.imwrite("test_result/%s/%s_result.jpg"%(model,img_path.split('/')[-1][:-4]),np.uint8(np.maximum(np.minimum(out_all[0,:,:,:] * 255.0,255.0),0.0)))    
    imageio.imwrite("test_result/%s/%s_input.jpg"%(model,img_path.split('/')[-1][:-4]),np.uint8(np.maximum(np.minimum(im[0,:,:,0] * 255.0,255.0),0.0)))    
