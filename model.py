from collections import namedtuple
import keras
from keras.layers import *
from keras.models import Model

monodepth_parameters = namedtuple('parameters', 
                        # 'encoder, '
                        'height, width, '
                        'batch_size, '
                        # 'num_threads, '
                        'num_epochs, '
                        'learning_rate',
                        # 'do_stereo, '
                        # 'wrap_mode, '
                        # 'use_deconv, '
                        # 'alpha_image_loss, '
                        # 'disp_gradient_loss_weight, '
                        # 'lr_loss_weight, '
                        # 'full_summary')
                        )

def resblock(x,num_layers,num_blocks,batch_norm=False):
  for i in range(num_blocks):
    x=resconv(x,num_layers)
  if batch_norm:
    x=BatchNormalization()(x)
  # x=MaxPool2D()(x)
  x=Conv2D(filters=num_layers,kernel_size=(1,1),strides=(2,2),activation='elu',padding='same')(x)
  return x

def resconv(x,num_layers):
  do_proj=x.shape[3]!=num_layers
  shortcut=[]
  x=Conv2D(filters=num_layers,kernel_size=(1,1),strides=(1,1),activation='elu',padding='same')(x)
  x=Conv2D(filters=num_layers,kernel_size=(3,3),strides=(1,1),activation='elu',padding='same')(x)
  x=Conv2D(filters=num_layers*4,kernel_size=(1,1),strides=(1,1),activation='elu',padding='same')(x)
  if do_proj:
    shortcut=Conv2D(filters=num_layers*4,kernel_size=(1,1),strides=(1,1),activation='elu',padding='same')(x)
  else:
    shortcut=x
  x=Add()([x,shortcut])
  x=Activation('elu')(x)
  return x

def upconv(x,num_out_layers,kernel_size,batch_norm=False):
  x=UpSampling2D()(x)
  x=Conv2D(filters=num_out_layers,kernel_size=kernel_size,padding='same',activation='elu')(x)
  if batch_norm:
    x=BatchNormalization()(x)
  return x

def UNet(
    input_size, 
    num_outputs,
    num_levels,
    residual_blocks_size,
    filters_size_start,
    batch_norm=False,
    dropout=False,
    dropout_percent=0.5,
    ):
  
  inputs=Input((input_size))
  x=inputs
  # ENCODER
  skips=[]
  # H/2
  x=Conv2D(filters=filters_size_start, kernel_size=(7,7),strides=(2,2),padding='same')(x)

  skips.append(x)
  # H/4,/8,/16,32
  for i in range(num_levels):
    x=resblock(x,num_layers=filters_size_start*(2**i),num_blocks=residual_blocks_size[i],batch_norm=True)
    skips.append(x)
 
  # DECODER
  for i in range(num_levels):
    x=upconv(x,num_out_layers=filters_size_start*(2**(num_levels-i-1)),kernel_size=(3,3),batch_norm=True)
    skip=skips[num_levels-i-1]
    skip=Conv2D(filters=filters_size_start*(2**(num_levels-i-1)),kernel_size=(1,1),
             padding='same',activation='elu')(skip)
    x=concatenate([skip,x],axis=3)
    x=Conv2D(filters=filters_size_start*(2**(num_levels-i-1)),kernel_size=(3,3),
             padding='same',activation='elu')(x)
    # print('Decoder out',x.shape)

  x=UpSampling2D()(x)
  activation='sigmoid'
  outputs=Conv2D(filters=num_outputs,kernel_size=(3,3),padding='same',activation=activation)(x)
  # print('FINAL SHAPE',x.shape)
    
  model=Model(inputs,outputs)
  return model