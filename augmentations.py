import tensorflow as tf
import random
from random import uniform
import numpy as np


"""
    Random augmentations for single image
"""
# Random shift brightness
def random_shift_brightness1(img):
    random_brightness = random.uniform(0.5, 2.0)
    img = img * random_brightness
    return img

# Random shift gamma
def random_shift_gamma1(img):
    random_gamma = random.uniform(0.8, 1.2)
    img  = img  ** random_gamma
    return img

# Random shift color
def random_shift_color1(img):
    white = tf.ones([tf.shape(img)[0], tf.shape(img)[1]])
    color_image = tf.stack([white * random.uniform(0.8,1.2) for i in range(3)], axis=2)
    img *= color_image
    return img

# Random saturation
def random_saturation1(img):
    img  = tf.clip_by_value(img,  0, 1)
    return img

# Random scale
def random_scale1(img):
    random_scale=random.uniform(1.0,2.0)
    h,w=img.shape[0],img.shape[1]
    img=tf.image.resize(img,[int(h*random_scale),int(w*random_scale)])
    img=tf.image.random_crop(img,[h,w,3])
    return img

# Random flip 
def random_flip1(img):
    do_flip=random.uniform(0, 1)
    if do_flip>0.5:
        img=tf.image.flip_left_right(img)
    return img

"""
    Random augmentations for image pairs
"""
# Random shift brightness
def random_shift_brightness2(left,right):
    random_brightness = random.uniform(0.5, 2.0)
    left = left * random_brightness
    right = right * random_brightness
    return left,right

# Random shift gamma
def random_shift_gamma2(left,right):
    random_gamma = random.uniform(0.8, 1.2)
    left  = left  ** random_gamma
    right = right ** random_gamma
    return left,right

# Random shift color
def random_shift_color2(left,right):
    white = tf.ones([tf.shape(left)[0], tf.shape(left)[1]])
    color_image = tf.stack([white * random.uniform(0.8,1.2) for i in range(3)], axis=2)
    left *= color_image
    right *= color_image
    return left,right

# Random saturation
def random_saturation2(left,right):
    left  = tf.clip_by_value(left,  0, 1)
    right = tf.clip_by_value(right, 0, 1)
    return left,right

# Random scale
def random_scale2(left,right):
    random_scale=random.uniform(1.0,2.0)
    h,w=left.shape[0],left.shape[1]
    combined=tf.concat([left,right],axis=-1)
    combined=tf.image.resize(combined,[int(h*random_scale),int(w*random_scale)])
    combined=tf.image.random_crop(combined,[h,w,6])
    left=combined[:,:,:3]
    right=combined[:,:,3:]
    return left,right

# Random flip 
def random_flip2(left,right):
    do_flip=random.uniform(0, 1)
    if do_flip>0.5:
        left_init=left
        left=tf.image.flip_left_right(right)
        right=tf.image.flip_left_right(left_init)
    return left,right

    