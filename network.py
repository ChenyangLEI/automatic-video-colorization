from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os,time,cv2
from scipy import io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import subprocess
import utils as utils


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

def bottleneck_block(input, channel=32, ext='g_conv'):
    conv1_1=slim.conv2d(input, channel // 4, [1,1], rate=1, activation_fn=lrelu, scope=ext +'_1_1')
    conv1_2=slim.conv2d(conv1_1, channel // 4, [3,3], rate=1, activation_fn=lrelu, scope=ext +'_1_2')
    conv1_3=slim.conv2d(conv1_2, channel, [1,1], rate=1, activation_fn=lrelu, scope=ext +'_1_3')
    return conv1_3

def bottleneck_unet(input, channel=32, output_channel=3,reuse=False,ext="",div_num=1):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    conv1=slim.conv2d(input,channel,[1,1], rate=1, activation_fn=lrelu, scope=ext+'g_conv1_1')
    conv1=slim.conv2d(conv1,channel,[3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv1_2')
    pool1=slim.max_pool2d(conv1, [2, 2], padding='SAME' )

    conv2 = bottleneck_block(pool1, channel * 2, ext='g_conv2')
    pool2=slim.max_pool2d(conv2, [2, 2], padding='SAME' )
    
    conv3 = bottleneck_block(pool2, channel * 4, ext='g_conv3')
    pool3=slim.max_pool2d(conv3, [2, 2], padding='SAME' )
    
    conv4 = bottleneck_block(pool3, channel * 8, ext='g_conv4')
    pool4=slim.max_pool2d(conv4, [2, 2], padding='SAME' )
    
    conv5 = bottleneck_block(pool4, channel * 16, ext='g_conv5')
    
    up6 =  bilinear_up_and_concat( conv5, conv4, channel*8, channel*16, scope=ext+"g_up_1" )
    conv6 = bottleneck_block(up6, channel * 8, ext='g_conv6')

    up7 =  bilinear_up_and_concat( conv6, conv3, channel*4, channel*8, scope=ext+"g_up_2" )
    conv7 = bottleneck_block(up7, channel * 4, ext='g_conv7')

    up8 =  bilinear_up_and_concat( conv7, conv2, channel*2, channel*4, scope=ext+"g_up_3" )
    conv8 = bottleneck_block(up8, channel * 2, ext='g_conv8')

    up9 =  bilinear_up_and_concat( conv8, conv1, channel, channel*2, scope=ext+"g_up_4" )

    conv9=slim.conv2d(up9,  channel,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv9_1')
    conv9=slim.conv2d(conv9,output_channel*div_num,[3,3], rate=1, activation_fn=None,  weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv9_2')
    return conv9


def dowmsample_unet(input, channel=32, output_channel=3, reuse=False, ext="",div_num=1):
    """
    docstring
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()

    down_input = tf.image.resize_images(input, [tf.shape(input)[1] // 2, tf.shape(input)[2] // 2])
    down_hyper = utils.build(tf.tile(down_input, [1,1,1,3]))

    conv0=slim.conv2d(input, 32, [3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv0_1')
    conv0=slim.conv2d(conv0, 32, [3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv0_2')
    pool0=slim.max_pool2d(conv0, [2, 2], padding='SAME' )

    net_input = tf.concat([pool0, down_hyper], axis=3)
    net_output =  VCN(net_input, output_channel = 64, reuse=reuse, div_num=div_num)


    up10 =  bilinear_up_and_concat(net_output, conv0, channel, channel*2, scope=ext+"g_up_5" )
    conv10=slim.conv2d(up10,  channel, [3,3], rate=1, activation_fn=lrelu, scope=ext+'g_conv10_1')
    conv10=slim.conv2d(conv10, output_channel * div_num, [3,3], rate=1, activation_fn=None, scope=ext+'g_conv10_2')
    return conv10

def plain_unet(input, channel=32, output_channel=3, reuse=False,ext="", div_num=1):
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
    conv9=slim.conv2d(conv9, output_channel*div_num, [3,3], rate=1, activation_fn=None,  weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv9_2')
    return conv9

def hyper_unet(input, channel=32, output_channel=3, reuse=False,ext="", div_num=1):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    input = utils.build(tf.tile(input, [1,1,1,3]))
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
    conv9=slim.conv2d(conv9, output_channel*div_num, [3,3], rate=1, activation_fn=None,  weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv9_2')
    return conv9


def VCN(input, channel=32, output_channel=3, reuse=False,ext="", div_num=1):
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
    conv9=slim.conv2d(conv9, output_channel*div_num, [3,3], rate=1, activation_fn=None,  weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'g_conv9_2')
    return conv9

def VCRN(input, channel=32, output_channel=3,reuse=False,ext="VCRN"):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    conv1=slim.conv2d(input,channel,[1,1], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'r_conv1_1')
    conv1=slim.conv2d(conv1,channel,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'r_conv1_2')
    pool1=slim.max_pool2d(conv1, [2, 2], padding='SAME' )
    conv2=slim.conv2d(pool1,channel*2,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'r_conv2_1')
    conv2=slim.conv2d(conv2,channel*2,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'r_conv2_2')
    pool2=slim.max_pool2d(conv2, [2, 2], padding='SAME' )
    conv3=slim.conv2d(pool2,channel*4,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'r_conv3_1')
    conv3=slim.conv2d(conv3,channel*4,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'r_conv3_2')
    pool3=slim.max_pool2d(conv3, [2, 2], padding='SAME' )
    conv4=slim.conv2d(pool3,channel*8,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'r_conv4_1')
    conv4=slim.conv2d(conv4,channel*8,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'r_conv4_2')
    pool4=slim.max_pool2d(conv4, [2, 2], padding='SAME' )
    conv5=slim.conv2d(pool4,channel*16,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'r_conv5_1')
    conv5=slim.conv2d(conv5,channel*16,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'r_conv5_2')
    up6 =  bilinear_up_and_concat( conv5, conv4, channel*8, channel*16, scope=ext+"r_up_1" )
    conv6=slim.conv2d(up6,  channel*8,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'r_conv6_1')
    conv6=slim.conv2d(conv6,channel*8,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'r_conv6_2')
    up7 =  bilinear_up_and_concat( conv6, conv3, channel*4, channel*8, scope=ext+"r_up_2" )
    conv7=slim.conv2d(up7,  channel*4,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'r_conv7_1')
    conv7=slim.conv2d(conv7,channel*4,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'r_conv7_2')
    up8 =  bilinear_up_and_concat( conv7, conv2, channel*2, channel*4, scope=ext+"r_up_3" )
    conv8=slim.conv2d(up8,  channel*2,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'r_conv8_1')
    conv8=slim.conv2d(conv8,channel*2,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'r_conv8_2')
    up9 =  bilinear_up_and_concat( conv8, conv1, channel, channel*2, scope=ext+"r_up_4" )
    conv9=slim.conv2d(up9,  channel,[3,3], rate=1, activation_fn=lrelu, weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'r_conv9_1')
    conv9=slim.conv2d(conv9,output_channel,[3,3], rate=1, activation_fn=None,  weights_initializer=tf.contrib.layers.xavier_initializer(),scope=ext+'r_conv9_2')
    return conv9


