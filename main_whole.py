#tensorflow 1.2.0 is needed
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os,time,cv2,scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import utils as utils
import myflowlib as flowlib
import flow_warp as flow_warp_op
import scipy.misc as sic
import subprocess
import network as net
import loss as loss
import argparse
from sklearn.neighbors import NearestNeighbors

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='Result_whole', type=str, help="Model Name")
parser.add_argument("--div_num", default=4, type=int, help="diverse num")
parser.add_argument("--save_freq", default=1, type=int, help="save frequency")
parser.add_argument("--test_dir", default='./data/test/JPEGImages/480p/cows', type=str, help="Test dir path")
parser.add_argument("--train_root", default="./data/train/JPEGImages/480p/", type=str, help="Test dir path")
parser.add_argument("--test_root", default="./data/test/JPEGImages/480p/", type=str, help="Test dir path")
parser.add_argument("--imgs_dir", default='../data/Imagenet', type=str, help="Test dir path")
parser.add_argument("--is_training", default=1, type=int, help="Training or test")
parser.add_argument("--continue_training", default=1, type=int, help="Restore checkpoint")


ARGS = parser.parse_args()
print(ARGS)

model=ARGS.model
div_num=ARGS.div_num
save_freq = ARGS.save_freq
test_dir = ARGS.test_dir
train_root = [ARGS.train_root]
test_root = [ARGS.test_root]
is_training=ARGS.is_training
continue_training=ARGS.continue_training
imgs_dir = ARGS.imgs_dir
num_frame = 2 # number of read in frames


os.environ["CUDA_VISIBLE_DEVICES"]=str(np.argmax( [int(x.split()[2]) for x in subprocess.Popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]))


def occlusion_mask(im0, im1, flow10):
    warp_im0 = flow_warp_op.flow_warp(im0, flow10)
    diff = tf.abs(im1 - warp_im0)
    mask = tf.reduce_max(diff, axis=3, keep_dims=True)
    mask = tf.less(mask, 0.05)
    mask = tf.cast(mask, tf.float32)
    mask = tf.tile(mask, [1,1,1,3])
    return mask, warp_im0

def Bilateral_NN(color_image, neigh):
    h,w = color_image.shape[:2]    
    color_image_X = np.tile(np.arange(w),[h,1])/500.
    color_image_Y = np.tile(np.arange(h),[w,1]).T/500.
    color_image_all = np.concatenate([color_image,color_image_X[:,:,np.newaxis], color_image_Y[:,:,np.newaxis]],axis=2)
    neigh.fit(np.reshape(color_image_all, [-1, 5]))
    idxs = neigh.kneighbors(np.reshape(color_image_all, [-1, 5]), 5, return_distance=False)
#    idxs = np.zeros([h*w, 5],dtype="int32")
    return idxs

def prepare_input_w_flow(path, num_frames,gray=False):
    file_id=os.path.splitext(os.path.basename(path))[0]
    input_image_src, input_image_target = utils.read_image_sequence(path, num_frames=num_frame)
    if input_image_target is None:
        return None, None, None, None
    if not gray:
        input_flow_forward,input_flow_backward = utils.read_flow_sequence(path.replace("JPEGImages","FLOWImages"), num_frames=num_frame)
    else:
        input_flow_forward,input_flow_backward = utils.read_flow_sequence(path.replace("JPEGImages","FLOWImages_GRAY"), num_frames=num_frame)        
    h=input_image_src.shape[0]//32*32
    w=input_image_src.shape[1]//32*32
    return np.float32(np.expand_dims(input_image_src[:h:2,:w:2,:],axis=0)),\
        np.float32(np.expand_dims(input_image_target[:h:2,:w:2,:],axis=0)),\
        np.expand_dims(input_flow_forward[:h:2,:w:2,:],axis=0)/2.0,\
        np.expand_dims(input_flow_backward[:h:2,:w:2,:],axis=0)/2.0

config=tf.ConfigProto()
config.gpu_options.allow_growth=True
sess=tf.Session(config=config)
train_low=utils.read_image_path(train_root)
test_low=utils.read_image_path(test_root)


input_idx=tf.placeholder(tf.int32,shape=[None,5*num_frame])
input_i=tf.placeholder(tf.float32,shape=[None,None,None,1*num_frame])
input_target=tf.placeholder(tf.float32,shape=[None,None,None,3*num_frame])
input_flow_forward=tf.placeholder(tf.float32,shape=[None,None,None,2*(num_frame-1)])
input_flow_backward=tf.placeholder(tf.float32,shape=[None,None,None,2*(num_frame-1)])


gray_flow_forward=tf.placeholder(tf.float32,shape=[None,None,None,2*(num_frame-1)])
gray_flow_backward=tf.placeholder(tf.float32,shape=[None,None,None,2*(num_frame-1)])
c0=tf.placeholder(tf.float32,shape=[None,None,None,3])
c1=tf.placeholder(tf.float32,shape=[None,None,None,3])


lossDict = {}
objDict={} 

#   X0, X1: Gray frames
#   Y0, Y1: Ground truth color frames
#   C0, C1: Colorized frames
with tf.variable_scope(tf.get_variable_scope()):
    X0, X1 = input_i[:,:,:,0:1], input_i[:,:,:,1:2]
    Y0, Y1 = input_target[:,:,:,0:3], input_target[:,:,:,3:6]
    with tf.variable_scope('individual'):
        C0=net.VCN(utils.build(tf.tile(X0, [1,1,1,3])),reuse=False)
        C1=net.VCN(utils.build(tf.tile(X1, [1,1,1,3])),reuse=True)        

    objDict["mask"],_=occlusion_mask(Y0,Y1,input_flow_backward[:,:,:,0:2])
    objDict["warped"]=flow_warp_op.flow_warp(C0,input_flow_backward[:,:,:,0:2])

    lossDict["RankDiv_im1"]=loss.RankDiverse_loss(C0, tf.tile(input_target[:,:,:,0:3], [1,1,1,div_num]),div_num)
    lossDict["RankDiv_im2"]=loss.RankDiverse_loss(C1, tf.tile(input_target[:,:,:,3:6], [1,1,1,div_num]),div_num)
    lossDict["RankDiv"]=lossDict["RankDiv_im1"]+lossDict["RankDiv_im2"]

    lossDict['Bilateral_im1']= sum([loss.KNN_loss(C0[:,:,:,3*i:3*i+3], input_idx[:,0:5]) for i in range(4)])
    lossDict['Bilateral_im2']= sum([loss.KNN_loss(C1[:,:,:,3*i:3*i+3], input_idx[:,5:10])for i in range(4)])
    lossDict['Bilateral']= lossDict['Bilateral_im2'] + lossDict['Bilateral_im1']

    lossDict["temporal"]=tf.reduce_mean(tf.multiply(tf.abs(objDict["warped"]-C1),tf.tile(objDict["mask"],[1,1,1,4])))*5


    lossDict["total"]=lossDict["RankDiv"]+lossDict["temporal"]#+lossDict['Bilateral']

    objDict["prediction_0"]=tf.concat([C0[:,:,:,0:3],C0[:,:,:,9:12],C0[:,:,:,3:6],C0[:,:,:,6:9]],axis=2)
    objDict["prediction_1"]=tf.concat([C1[:,:,:,0:3],C1[:,:,:,9:12],C1[:,:,:,3:6],C1[:,:,:,6:9]],axis=2)


    #-------------RefineNet---------------#
    cmap_C, warp_C0 = occlusion_mask(c0, c1, gray_flow_backward[:,:,:,0:2])
    cmap_X, warp_X0 = occlusion_mask(tf.tile(input_i[:,:,:,0:1], [1,1,1,3]), tf.tile(input_i[:,:,:,1:2],[1,1,1,3]),gray_flow_backward[:,:,:,0:2])
    low_conf_mask = tf.cast(tf.greater(cmap_X - cmap_C, 0), tf.float32)
   
    coarse_C1 = c1*(-low_conf_mask+1) + tf.tile(input_i[:,:,:,1:2],[1,1,1,3])*low_conf_mask
    ref_input = tf.concat([coarse_C1, warp_C0, c1, low_conf_mask, cmap_C, cmap_X], axis=3)
    # ref_input = tf.concat([warp_C0, c1, cmap_C, cmap_X, low_conf_mask], axis=3)
    final_r1 = net.VCRN(ref_input)
    # temporal_g_loss = tf.reduce_mean(tf.abs(input_target[:,:,:,3:6]-final_r1)) + tf.reduce_mean(tf.multiply(tf.abs(c1-final_r1),-low_conf_mask+1))
    temporal_g_loss = tf.reduce_mean(tf.abs(input_target[:,:,:,3:6]-final_r1)) + \
                  tf.reduce_mean(tf.multiply(tf.abs(warp_C0-final_r1),low_conf_mask)) + \
                  tf.reduce_mean(tf.multiply(tf.abs(c1-final_r1),-low_conf_mask+1))

opt=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(lossDict["total"],var_list=[var for var in tf.trainable_variables()])
opt2=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(0.2*lossDict["RankDiv_im1"],var_list=[var for var in tf.trainable_variables()])
opt_refine=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(temporal_g_loss,var_list=[var for var in tf.trainable_variables() if var.name.startswith('VCRN')])


print([var for var in tf.trainable_variables() if var.name.startswith('VCRN')])

saver=tf.train.Saver(max_to_keep=1000)
sess.run([tf.global_variables_initializer()])

var_restore = [v for v in tf.trainable_variables()]
saver_restore=tf.train.Saver(var_restore)
ckpt=tf.train.get_checkpoint_state(model)
print("contain checkpoint: ", ckpt)
if ckpt and continue_training:
    print('loaded '+ ckpt.model_checkpoint_path)
    saver_restore.restore(sess,ckpt.model_checkpoint_path)

neigh=NearestNeighbors(n_neighbors=5)
maxepoch=1001
num_train=len(train_low)
print("Number of training images: ", num_train)

print("is_training: ", is_training )
if is_training:
    for epoch in range(1,maxepoch):
        print("Processing epoch %d"%epoch)
        input_list_src=[None]*num_train
        input_list_target=[None]*num_train
        input_list_flow_forward=[None]*num_train
        input_list_flow_backward=[None]*num_train
        gray_list_flow_forward=[None]*num_train   
        gray_list_flow_backward=[None]*num_train
        if os.path.isdir("%s/%04d"%(model,epoch)):
            continue
        cnt=0
        all_RD,all_Bi = 0., 0.
        #Images
        for id in np.random.permutation(1431167):#1431167
            st=time.time()
            color_image=np.float32(scipy.misc.imread("%s/%06d.jpg"%(imgs_dir, id+1)))/255.0
            if len(color_image.shape)==2:
                continue
            h=color_image.shape[0]//32*32
            w=color_image.shape[1]//32*32            
            color_image=color_image[:h:2,:w:2,:]
            gray_image=cv2.cvtColor(color_image,cv2.COLOR_RGB2GRAY)
            if gray_image is None:
                print(id)
                continue
            idxs = Bilateral_NN(color_image, neigh)
            gray_image=gray_image[:,:,np.newaxis]

            _, crt_RDLoss, crt_BiLoss = sess.run([opt2,lossDict["RankDiv_im1"], lossDict["Bilateral_im1"]],feed_dict={input_i:gray_image[np.newaxis,:,:,[0,0]],\
                input_target:color_image[np.newaxis,:,:,[0,1,2,0,1,2]], input_idx:np.tile(idxs,[1,2])})
            cnt+=1
            all_RD += crt_RDLoss
            all_Bi += crt_BiLoss
            print("Image iter: %d %d || RankDiv: %.4f %.4f|| Bi: %.4f %.4f || Time: %.4f"%(epoch,cnt,crt_RDLoss,all_RD/cnt,crt_BiLoss,all_Bi/cnt,time.time()-st))
            if cnt>=5000:
                break

        # Video VCN
        cnt=0
        all_D1, all_D2, all_B1, all_B2, all_T, all_loss = 0,0,0,0,0,0
        for id in np.random.permutation(num_train):
            st=time.time()
            if input_list_src[id] is None:
                input_list_src[id], input_list_target[id], input_list_flow_forward[id], input_list_flow_backward[id] = prepare_input_w_flow(train_low[id], num_frames=num_frame)
            if input_list_src[id] is None or input_list_target[id] is None or input_list_flow_forward[id] is None:
                continue
            input_frames_processed = input_list_src[id]

            idxs1 = Bilateral_NN(input_list_target[id][0,:,:,:3], neigh)
            idxs2 = Bilateral_NN(input_list_target[id][0,:,:,3:6], neigh)

            _, out_loss, C0_im, C1_im = sess.run([opt, lossDict,C0,C1],\
                feed_dict={input_i:input_list_src[id],input_target:input_list_target[id],\
                           input_flow_backward:input_list_flow_backward[id],\
                           input_idx: np.concatenate([idxs1,idxs2],axis=1)})
            all_D1 += out_loss["RankDiv_im1"]
            all_D2 += out_loss["RankDiv_im2"]
            all_B1 += out_loss["Bilateral_im1"]
            all_B2 += out_loss["Bilateral_im2"]
            all_T  += out_loss["temporal"]
            all_loss += out_loss["total"]

            cnt+=1
            print("iter: %d %d %.2fs loss: %.4f %.4f|| (D1) %.4f %.4f (D2) %.4f %.4f || (B1) %.4f %.4f (B2) %.4f %.4f (T) %.4f %.4f"\
                %(epoch,cnt,out_loss["total"],all_loss/cnt, time.time()-st,\
                    out_loss["RankDiv_im1"], all_D1/cnt, out_loss["RankDiv_im2"], all_D2/cnt,\
                    out_loss["Bilateral_im1"], all_B1/cnt, out_loss["Bilateral_im2"], all_B2/cnt,\
                    out_loss["temporal"], all_T/cnt))

            # Video Refine
            if epoch > 0:
                _, _, gray_list_flow_forward[id], gray_list_flow_backward[id] = prepare_input_w_flow(train_low[id], num_frames=num_frame)
                _, out_loss, final_C1 = sess.run([opt_refine, temporal_g_loss, final_r1],\
                        feed_dict={c0:C0_im[:,:,:,0:3], c1: C1_im[:,:,:,0:3], input_i:input_list_src[id],input_target:input_list_target[id],\
                           input_flow_backward:input_list_flow_backward[id],\
                           gray_flow_backward:gray_list_flow_backward[id],\
                           input_idx: np.concatenate([idxs1,idxs2],axis=1)})
                print("iter: %d %d || Refine || loss: %.4f %.4f"%(epoch,cnt,out_loss,time.time()-st))

            if cnt>=1000:
                break

        # Validation
        if not os.path.isdir("%s/%04d"%(model,epoch)):
            os.makedirs("%s/%04d"%(model,epoch))

        if epoch % save_freq == 0:
            numtest=len(test_low)
            all_loss_test=np.zeros(numtest, dtype=float)
            for ind in range(numtest):
                if ind>30 and epoch%25>0:
                    break
                input_image_src, input_image_target, input_flow_forward_src, input_flow_backward_src = prepare_input_w_flow(test_low[int(ind*60/pow(60,int(epoch%25==0)))],num_frames=num_frame)
                if input_image_src is None or input_image_target is None or input_flow_forward_src is None:
                    print("Not able to read the images/flows.")
                    flag=True
                    continue
                st=time.time()
                C0_imall,C1_imall,C0_im, C1_im, warped,mask=sess.run([objDict["prediction_0"],objDict["prediction_1"],C0, C1,objDict["warped"],objDict['mask']],feed_dict={input_i:input_image_src,
                    input_target:input_image_target,
                    input_flow_backward:input_flow_backward_src
                    })
                print("test time for %s --> %.3f"%(ind, time.time()-st))
                input_image_src, input_image_target, gray_flow_forward_src, gray_flow_backward_src = prepare_input_w_flow(test_low[int(ind*60/pow(60,int(epoch%25==0)))],num_frames=num_frame)
                h,w = C0_im.shape[1:3]
                outputs= []
                for ref_i in range(4):
                    output, out_cmap_C, out_cmap_X, out_low_conf_mask = sess.run([final_r1, cmap_C,cmap_X,low_conf_mask],feed_dict={c0:C0_im[:,:,:,ref_i*3:ref_i*3+3], c1:C1_im[:,:,:,ref_i*3:ref_i*3+3], \
                           input_i:input_image_src, input_target:input_image_target, \
                           gray_flow_backward:gray_flow_backward_src, input_flow_backward:input_flow_backward_src})

                    outputs.append(output[0,:,:,:])
                    # Debug

                if not os.path.isdir("%s/%04d/predictions" % (model, epoch)):
                    os.makedirs("%s/%04d/predictions" % (model, epoch))
                sic.imsave("%s/%04d/predictions/mask_%06d.jpg"%(model, epoch, ind),np.uint8(np.maximum(np.minimum(np.concatenate([out_cmap_C[0],out_cmap_X[0],out_low_conf_mask[0]],axis=1)* 255.0,255.0),0.0)))
                sic.imsave("%s/%04d/predictions/final_%06d.jpg"%(model, epoch, ind),np.uint8(np.maximum(np.minimum(output[0] * 255.0,255.0),0.0)))
                sic.imsave("%s/%04d/predictions/predictions_%06d.jpg"%(model, epoch, ind),np.uint8(np.maximum(np.minimum(C0_imall[0] * 255.0,255.0),0.0)))
                
                saver.save(sess,"%s/model.ckpt"%model)
            if epoch%10==0:
               saver.save(sess,"%s/%04d/model.ckpt"%(model,epoch))

# Inference
else:
    test_low=utils.get_names(test_dir)
    numtest=len(test_low)
    print(test_low[0])
    out_folder = test_dir.split('/')[-1]
    outputs= [None]*4
    for ind in range(numtest):
        input_image_src, input_image_target, input_flow_forward_src, input_flow_backward_src = prepare_input_w_flow(test_low[ind],num_frames=num_frame,gray=True)
        if input_image_src is None or input_image_target is None or input_flow_forward_src is None:
            print("Not able to read the images/flows.")
            continue
        st=time.time()
        C0_imall,C1_imall,C0_im, C1_im=sess.run([objDict["prediction_0"],objDict["prediction_1"],C0, C1],feed_dict={input_i:input_image_src,
            input_target:input_image_target,
            input_flow_backward:input_flow_backward_src
            })
        print("test time for %s --> %.3f"%(ind, time.time()-st))
        h,w = C0_im.shape[1:3]
        print(C0_im.shape)
        if not os.path.isdir("%s/%s" % (model, out_folder)):
            os.makedirs("%s/%s/predictions" % (model, out_folder))
            os.makedirs("%s/%s/predictions0" % (model, out_folder))
            os.makedirs("%s/%s/predictions1" % (model, out_folder))
            os.makedirs("%s/%s/predictions2" % (model, out_folder))
            os.makedirs("%s/%s/predictions3" % (model, out_folder))
        if ind == 0:
            for ref_i in range(4):
                output,_ = sess.run([final_r1, temporal_g_loss],feed_dict={c0:C0_im[:,:,:,ref_i*3:ref_i*3+3], c1:C1_im[:,:,:,ref_i*3:ref_i*3+3], \
                       input_i:input_image_src, input_target:input_image_target, \
                       gray_flow_backward:input_flow_backward_src, input_flow_backward:input_flow_backward_src})
                outputs[ref_i] = output
                sic.imsave("%s/%s/predictions%d/final_%06d.jpg"%(model, out_folder, ref_i, ind),np.uint8(np.maximum(np.minimum(C0_im[0,:,:,ref_i*3:ref_i*3+3] * 255.0,255.0),0.0)))
                sic.imsave("%s/%s/predictions%d/final_%06d.jpg"%(model, out_folder, ref_i, ind+1),np.uint8(np.maximum(np.minimum(output[0,:,:,:] * 255.0,255.0),0.0)))
            sic.imsave("%s/%s/predictions/predictions_%06d.jpg"%(model, out_folder, ind+1),np.uint8(np.maximum(np.minimum(C1_imall[0,:,:,:] * 255.0,255.0),0.0)))
            sic.imsave("%s/%s/predictions/final_%06d.jpg"%(model, out_folder, ind+1),np.uint8(np.maximum(np.minimum(np.concatenate(outputs,axis=2)[0,:,:,:] * 255.0,255.0),0.0)))

        else:
            for ref_i in range(4):
                output,_ = sess.run([final_r1, temporal_g_loss],feed_dict={c0:outputs[ref_i], c1:C1_im[:,:,:,:3], \
                       input_i:input_image_src, input_target:input_image_target, \
                       gray_flow_backward:input_flow_backward_src, input_flow_backward:input_flow_backward_src})
                outputs.append(output[0,:,:,:])
                sic.imsave("%s/%s/predictions%d/final_%06d.jpg"%(model, out_folder, ref_i, ind+1),np.uint8(np.maximum(np.minimum(output[0,:,:,:] * 255.0,255.0),0.0)))
            sic.imsave("%s/%s/predictions/predictions_%06d.jpg"%(model, out_folder, ind+1),np.uint8(np.maximum(np.minimum(C1_imall[0,:,:,:] * 255.0,255.0),0.0)))
