import numpy as np
import keras.backend as K
import tensorflow as tf
import skimage.measure
import bilinear_sampler
import utils
import importlib
# importlib.reload(bilinear_sampler)



"""
    Appearance matching loss
"""
# L1
def l1_loss(y_true,y_pred):
    return K.sum(K.abs(y_true-y_pred))

# SSIM
def ssim(x,y):
    # return skimage.measure.compare_ssim(x.numpy(),y.numpy(),multichannel=True)
    # print('shapes in ssim',x.shape)
    return tf.image.ssim(x,y,max_val=1.0)

def appearance_matching_loss_single(
    alpha_image_loss,
    l_true,
    l_recons,
    r_true,
    r_recons):

    l_loss=alpha_image_loss * ((1-ssim(l_true,l_recons))/2) + (1-alpha_image_loss) * l1_loss(l_true,l_recons)
    r_loss=alpha_image_loss * ((1-ssim(r_true,r_recons))/2) + (1-alpha_image_loss) * l1_loss(r_true,r_recons)

    N=1024*2048

    return (l_loss+r_loss)/N

@tf.function
def appearance_matching_loss(
    alpha_image_loss,
    l_true,
    l_recons,
    r_true,
    r_recons):

    # losses=[]
    losses=tf.zeros([1],tf.float32)
    for i in range(len(l_true)):
        loss=appearance_matching_loss_single(alpha_image_loss, l_true[i], l_recons[i],r_true[i],r_recons[i])
        losses=tf.math.add(losses,loss)
        # losses.append(loss.numpy())
        print('loss',loss)
    # return K.mean(losses)
    return losses

"""
    Disparity smoothness loss
"""
def comp_disp_gradient_x(img):
    h,w=1024,2048
    grad=(img[:,:-1,:] - img[:,1:,:])
    zero_col=tf.zeros((1024,1,1),dtype=tf.dtypes.float32)
    return tf.concat((grad,zero_col),axis=1)

def comp_img_gradient_x(img):
    h,w=1024,2048
    grad=(img[:,:-1,:] - img[:,1:,:])
    zero_col=tf.zeros((1024,1,3),dtype=tf.dtypes.float32)
    return tf.concat((grad,zero_col),axis=1)

def comp_disp_gradient_y(img):
    h,w=1024,2048
    grad=img[:-1,:,:] - img[1:,:,:]
    zero_col=tf.zeros((1,2048,1),dtype=tf.dtypes.float32)
    return tf.concat((grad,zero_col),axis=0)

def comp_img_gradient_y(img):
    h,w=1024,2048
    grad=img[:-1,:,:] - img[1:,:,:]
    zero_col=tf.zeros((1,2048,3),dtype=tf.dtypes.float32)
    return tf.concat((grad,zero_col),axis=0)

def get_disparity_smoothness(img,disp):
    disp_gradient_x=comp_disp_gradient_x(disp)
    disp_gradient_y=comp_disp_gradient_y(disp)

    img_gradient_x=comp_img_gradient_x(img)
    img_gradient_y=comp_img_gradient_y(img)

    weight_x=np.exp(-np.abs(img_gradient_x))
    weight_y=np.exp(-np.abs(img_gradient_y))

    smoothness_x=np.abs(disp_gradient_x) * weight_x
    smoothness_y=np.abs(disp_gradient_y) * weight_y

    N=1024*2048
    return (np.sum(smoothness_x+smoothness_y))/N

def disparity_smoothness_loss_single(
    l_true,
    l_disp_pred,
    r_true,
    r_disp_pred):

    l_disp=get_disparity_smoothness(l_true,l_disp_pred)
    r_disp=get_disparity_smoothness(r_true,r_disp_pred)

    return l_disp+r_disp

def disparity_smoothness_loss(
    l_true,
    l_disp_pred,
    r_true,
    r_disp_pred):
    
    losses=[]
    for i in range(len(l_true)):
        loss=disparity_smoothness_loss_single(l_true[i], l_disp_pred[i],r_true[i],r_disp_pred[i])
        losses.append(loss)
    return np.mean(losses)

"""
    Left-Right disparity consistency losss
"""

def lr_disparity_consistency_loss(
    l_disp,
    r_disp):

    N=l_disp[0].shape[0]*l_disp[0].shape[1]

    l_disp_neg=[-x for x in l_disp]
    r_to_l_disp=bilinear_sampler.billinear_sampler_fct(r_disp,l_disp_neg)
    l_to_r_disp=bilinear_sampler.billinear_sampler_fct(l_disp,r_disp)

    losses=[]
    for i in range(len(l_disp)):
        utils.visualize(pred_disp1=r_to_l_disp[i],pred_disp2=l_to_r_disp[i])
        l_loss=l1_loss(r_to_l_disp,l_disp)
        r_loss=l1_loss(l_to_r_disp,r_disp)
        loss=(l_loss+r_loss)/N
        losses.append(loss)
    return np.mean(losses)

"""
Total loss
"""
def total_loss(
    alpha_image_loss,
    disp_gradient_loss_weight,
    lr_loss_weight,
    l_true,
    l_recons,
    l_disp_pred,
    r_true,
    r_recons,
    r_disp_pred,
    ):

    # print('recons shape in total loss fct',l_recons.shape)
    # print('input shape in total loss fct',l_true.shape)
    # print('disparity shape in total loss fct',l_disp_pred.shape)
    Cap=appearance_matching_loss(alpha_image_loss,l_true,l_recons,r_true,r_recons)
    Cds=disparity_smoothness_loss(l_true,l_disp_pred,r_true,r_disp_pred)
    Clr=lr_disparity_consistency_loss(l_disp_pred,r_disp_pred)
    
    loss=alpha_image_loss*Cap + disp_gradient_loss_weight*Cds + lr_loss_weight*Clr
    return loss




