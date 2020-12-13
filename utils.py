import numpy as np
import tensorflow as tf
import os, cv2
import math
from scipy import io
import myflowlib as flowlib
import scipy.misc as sic
# from flownet2.src import flow_warp
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops


IMG_EXTENSIONS = [
    '.png', '.PNG', 'jpg', 'JPG', '.jpeg', '.JPEG',
    '.ppm', '.PPM', '.bmp', '.BMP',
]
def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError("image must be either 3 or 4 dimensions")

        # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image

def get_names(dir='./'):
    old_names = os.popen("ls %s"%dir).readlines()
    new_names = [None]*len(old_names)
    for idx in range(len(old_names)):
        new_names[idx] = dir+'/'+old_names[idx][:-1]
    return new_names


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

'''
Return: gray_images, color_images
        gray_images-    [num_frames, H, W, 1]
        color_images-   [num_frames, H, W, 3]
        pixel values [0,1]
'''
def read_image_sequence(filename, num_frames):
    file1 = os.path.splitext(os.path.basename(filename))[0]
    ext = os.path.splitext(os.path.basename(filename))[1]
    try:
        img1 = sic.imread(filename).astype(np.float32)
        imgh1 = img1
    except:
        #print("Cannot read the first frame.")
        return None, None
    if len(img1.shape) == 2: # discard grayscale images
        return None, None

    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img1 = np.expand_dims(img1,2)
    
    img_l_seq=img1/255.0
    img_h_seq=imgh1/255.0
    for i in range(num_frames-1):
        filei = int(file1) + i + 1
        filenamei = os.path.split(filename)[0] + "/" + "{:>05}".format(filei).format() + ext
        try:
            imgi = sic.imread(filenamei).astype(np.float32)
            imghi = imgi
        except:
            #print("Cannot read the following %d frames\n"%(num_frames))
            return None, None
        imgi = cv2.cvtColor(imgi, cv2.COLOR_RGB2GRAY)
        imgi = np.expand_dims(imgi,2)

        img_l_seq = np.concatenate((img_l_seq,imgi/255.0),axis=2)
        img_h_seq = np.concatenate((img_h_seq,imghi/255.0),axis=2)

    return img_l_seq, img_h_seq


def read_image_SPMC(filename, num_frames):
    file1 = os.path.splitext(os.path.basename(filename))[0]
    ext = os.path.splitext(os.path.basename(filename))[1]

    img1 = sic.imread(filename)
    imgh1 = sic.imread(filename.replace("input4","gt"))
    if img1 is None or imgh1 is None:
        print("Cannot read the first frame\n")
        return None,None
    if len(img1.shape) == 2:
        isgray=True
        img1 = np.concatenate((img1,img1,img1),axis=2)
        imgh1 = np.concatenate((imgh1,imgh1,imgh1),axis=2)
    else:
        isgray=False
    
    img_l_seq=img1
    img_h_seq=imgh1
    for i in range(num_frames-1):
        filei = int(file1) + i + 1
        filenamei = os.path.split(filename)[0] + "/" + "{:>04}".format(filei).format() + ext
        imgi = sic.imread(filenamei, -1)
        imghi = sic.imread(filenamei.replace("input4","gt"), -1)
        if imgi is None:
            print("Cannot read the following %d frames\n"%(num_frames))
            return None,None
        else:
            if isgray:
                imgi = np.concatenate((imgi,imgi,imgi),axis=2)
                imghi = np.concatenate((imghi,imghi,imghi),axis=2)
        img_l_seq = np.concatenate((img_l_seq,imgi),axis=2)
        img_h_seq = np.concatenate((img_h_seq,imghi),axis=2)

    return img_l_seq, img_h_seq

def read_flow_sequence_w_mask(filename, num_frames):
    file1 = os.path.splitext(os.path.basename(filename))[0]
    folder = os.path.split(filename)[0]
    ext = os.path.splitext(os.path.basename(filename))[1]
    
    filej = file1
    for i in range(num_frames-1):
        filei = int(file1) + i + 1
        if "SPMC" in filename:
            flow_forward = flowlib.read_flow(folder+"/Forward/{:>04}".format(filej).format()+"_"+"{:>04}".format(filei).format()+".flo")
            flow_forward_mask = sic.imread(folder+"/Forward/{:>04}".format(filej).format()+"_"+"{:>04}".format(filei).format()+".png")
            flow_backward = flowlib.read_flow(folder+"/Backward/{:>04}".format(filei).format()+"_"+"{:>04}".format(filej).format()+".flo")
            flow_backward_mask = sic.imread(folder+"/Forward/{:>04}".format(filej).format()+"_"+"{:>04}".format(filei).format()+".png")
        else:
            flow_forward = flowlib.read_flow(folder.replace("480p","Forward")+"/"+"{:>05}".format(filej).format()+"_"+"{:>05}".format(filei).format()+".flo")
            flow_forward_mask = sic.imread(folder.replace("480p","Forward")+"/"+"{:>05}".format(filej).format()+"_"+"{:>05}".format(filei).format()+".png")
            flow_backward = flowlib.read_flow(folder.replace("480p","Backward")+"/"+"{:>05}".format(filei).format()+"_"+"{:>05}".format(filej).format()+".flo")
            flow_backward_mask = sic.imread(folder.replace("480p","Backward")+"/"+"{:>05}".format(filei).format()+"_"+"{:>05}".format(filej).format()+".png")
        filej = filei
        if i == 0:
            flow_forward_seq = flow_forward
            flow_backward_seq = flow_backward
            flow_backward_mask_seq = np.expand_dims(flow_backward_mask, axis=2)
            flow_forward_mask_seq = np.expand_dims(flow_forward_mask, axis=2)
        else:
            flow_forward_seq = np.concatenate((flow_forward_seq, flow_forward), axis=2)
            flow_backward_seq = np.concatenate((flow_backward_seq, flow_backward), axis=2)
            flow_forward_mask_seq = np.concatenate((flow_forward_mask_seq, np.expand_dims(flow_forward_mask, axis=2)), axis=2)
            flow_backward_mask_seq = np.concatenate((flow_backward_mask_seq, np.expand_dims(flow_backward_mask, axis=2)), axis=2)

    return flow_forward_seq, flow_backward_seq, flow_forward_mask_seq, flow_backward_mask_seq




def read_flow_sequence(filename, num_frames):
    file1 = os.path.splitext(os.path.basename(filename))[0]
    folder = os.path.split(filename)[0]
    ext = os.path.splitext(os.path.basename(filename))[1]
    
    filej = file1
    for i in range(num_frames-1):
        filei = int(file1) + i + 1
        if "SPMC" in filename:
            flow_forward = flowlib.read_flow(folder+"/Forward/{:>04}".format(filej).format()+"_"+"{:>04}".format(filei).format()+".flo")
            flow_backward = flowlib.read_flow(folder+"/Backward/{:>04}".format(filei).format()+"_"+"{:>04}".format(filej).format()+".flo")
        else:
            flow_forward = flowlib.read_flow(folder.replace("480p","Forward")+"/"+"{:>05}".format(filej).format()+"_"+"{:>05}".format(filei).format()+".flo")
            flow_backward = flowlib.read_flow(folder.replace("480p","Backward")+"/"+"{:>05}".format(filei).format()+"_"+"{:>05}".format(filej).format()+".flo")
        filej = filei
        if i == 0:
            flow_forward_seq = flow_forward
            flow_backward_seq = flow_backward
        else:
            flow_forward_seq = np.concatenate((flow_forward_seq, flow_forward), axis=2)
            flow_backward_seq = np.concatenate((flow_backward_seq, flow_backward), axis=2)

    return flow_forward_seq, flow_backward_seq


def read_flow_sintel(filename, num_frames, substr="clean"):
    file1 = os.path.splitext(os.path.basename(filename))[0]
    folder = os.path.split(filename)[0]
    flowfolder = folder.replace(substr, "flow")
    maskfolder = folder.replace(substr, "occlusions")
    invalfolder = folder.replace(substr, "invalid")
    start_frame = int(file1.split("_")[1])
    for i in range(num_frames-1):
        # name = "/frame_" + "{:>04}".format(str(start_frame + i )).format() 
        name = "/frame_" + "{:>04}".format(str(start_frame + i - 1)).format()
        # print("flow file: ",name)
        try:
            flowi = flowlib.read_flow(flowfolder+name+".flo")
            flow_maski = sic.imread(maskfolder+name+".png",0).astype(np.float32)/255.
            flow_invali = sic.imread(invalfolder+name+".png",0).astype(np.float32)/255.
            
            flow_maski = np.expand_dims((flow_maski),axis=2) 
            flow_invali = np.expand_dims((flow_invali),axis=2)
        except:
            return None, None, None
        if i == 0:
            # initialize the sequence
            flow_seq = flowi
            flow_seq_mask = flow_maski
            flow_seq_inval = flow_invali
        else:
            flow_seq = np.concatenate((flow_seq, flowi),axis=2)
            flow_seq_mask = np.concatenate((flow_seq_mask, flow_maski), axis=2)
            flow_seq_inval = np.concatenate((flow_seq_inval, flow_invali), axis=2)
    return flow_seq, flow_seq_mask, flow_seq_inval

def read_image_path(file_path):
    path_all=[]
    for dirname in file_path:
        for root, dir, fnames in sorted(os.walk(dirname)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path_all.append(os.path.join(root, fname))
    return path_all

def flip_images(X):
    num_img = X.shape[0]
    magic=np.random.random()
    if magic < 0.3:
        for i in range(num_img):
            X[i,...] = np.fliplr(X[i,...])
    magic=np.random.random()
    if magic < 0.3:
        for i in range(num_img):
            X[i,...] = np.fliplr(X[i,...])
    return X

def pad_images(X,a):
    num_img = X.shape[0]
    h_orig,w_orig = X.shape[1:3]
    newX = np.ones((num_img,a,a,3))
    for i in range(num_img):
        pad_width=((0,a-h_orig),(0,a-w_orig),(0,0))
        newX[i,...] = np.pad(X[i,...],pad_width,'constant')
    return newX

def crop_images(X,a,b,is_sq=False):
    h_orig,w_orig = X.shape[1:3]
    w_crop = np.random.randint(a, b)
    r = w_crop/w_orig
    h_crop = np.int(h_orig*r)
    try:
        w_offset = np.random.randint(0, w_orig-w_crop-1)
        h_offset = np.random.randint(0, h_orig-h_crop-1)
    except:
        print("Original W %d, desired W %d"%(w_orig,w_crop))
        print("Original H %d, desired H %d"%(h_orig,h_crop))
    return X[:,h_offset:h_offset+h_crop-1,w_offset:w_offset+w_crop-1,:]

def degamma(X):
    return np.power(X, 2.2)

def gamma(X):
    return np.power(X, 1/2.2)

def build_net(ntype,nin,nwb=None,name=None):
    if ntype=='conv':
        return tf.nn.relu(tf.nn.conv2d(nin,nwb[0],strides=[1,1,1,1],padding='SAME',name=name)+nwb[1])
    elif ntype=='pool':
        return tf.nn.avg_pool(nin,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


def get_weight_bias(vgg_layers,i):
    weights=vgg_layers[i][0][0][2][0][0]
    weights=tf.constant(weights)
    bias=vgg_layers[i][0][0][2][0][1]
    bias=tf.constant(np.reshape(bias,(bias.size)))
    return weights,bias

vgg_rawnet=io.loadmat('VGG_Model/imagenet-vgg-verydeep-19.mat')
def build_vgg19(input,reuse=False):
    with tf.variable_scope("vgg19"):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        net={}
        vgg_layers=vgg_rawnet['layers'][0]
        net['input']=input-np.array([123.6800, 116.7790, 103.9390]).reshape((1,1,1,3))
        net['conv1_1']=build_net('conv',net['input'],get_weight_bias(vgg_layers,0),name='vgg_conv1_1')
        net['conv1_2']=build_net('conv',net['conv1_1'],get_weight_bias(vgg_layers,2),name='vgg_conv1_2')
        net['pool1']=build_net('pool',net['conv1_2'])
        net['conv2_1']=build_net('conv',net['pool1'],get_weight_bias(vgg_layers,5),name='vgg_conv2_1')
        net['conv2_2']=build_net('conv',net['conv2_1'],get_weight_bias(vgg_layers,7),name='vgg_conv2_2')
        net['pool2']=build_net('pool',net['conv2_2'])
        net['conv3_1']=build_net('conv',net['pool2'],get_weight_bias(vgg_layers,10),name='vgg_conv3_1')
        net['conv3_2']=build_net('conv',net['conv3_1'],get_weight_bias(vgg_layers,12),name='vgg_conv3_2')
        net['conv3_3']=build_net('conv',net['conv3_2'],get_weight_bias(vgg_layers,14),name='vgg_conv3_3')
        net['conv3_4']=build_net('conv',net['conv3_3'],get_weight_bias(vgg_layers,16),name='vgg_conv3_4')
        net['pool3']=build_net('pool',net['conv3_4'])
        net['conv4_1']=build_net('conv',net['pool3'],get_weight_bias(vgg_layers,19),name='vgg_conv4_1')
        net['conv4_2']=build_net('conv',net['conv4_1'],get_weight_bias(vgg_layers,21),name='vgg_conv4_2')
        net['conv4_3']=build_net('conv',net['conv4_2'],get_weight_bias(vgg_layers,23),name='vgg_conv4_3')
        net['conv4_4']=build_net('conv',net['conv4_3'],get_weight_bias(vgg_layers,25),name='vgg_conv4_4')
        net['pool4']=build_net('pool',net['conv4_4'])
        net['conv5_1']=build_net('conv',net['pool4'],get_weight_bias(vgg_layers,28),name='vgg_conv5_1')
        net['conv5_2']=build_net('conv',net['conv5_1'],get_weight_bias(vgg_layers,30),name='vgg_conv5_2')
        return net

def build(input):
    vgg19_features=build_vgg19(input[:,:,:,0:3]*255.0)
    for layer_id in range(1,6):#6
        vgg19_f = vgg19_features['conv%d_2'%layer_id]
        input = tf.concat([tf.image.resize_bilinear(vgg19_f,(tf.shape(input)[1],tf.shape(input)[2]))/255.0,input], axis=3)
    return input

def build_nlayer(input, nlayer):
    vgg19_features=build_vgg19(input[:,:,:,0:3]*255.0)
    for layer_id in range(1,nlayer):#6
        vgg19_f = vgg19_features['conv%d_2'%layer_id]
        input = tf.concat([tf.image.resize_bilinear(vgg19_f,(tf.shape(input)[1],tf.shape(input)[2]))/255.0,input], axis=3)
    return input


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
def conv_2d(input_, output_dim, 
           k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
           name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv
