import tensorflow as tf

"""
Repeat fct
"""
def repeat(x, n_repeats):
    rep = tf.tile(tf.expand_dims(x, 1), [1, int(n_repeats)])
    return tf.reshape(rep, [-1])

"""
Interpolate
"""
def interpolate(img,x,y,num_batch,height,width,num_channels):
    _height_f=float(height)
    _width_f=float(width)

    _edge_size=0

    x=tf.clip_by_value(x,0.0,_width_f-1+2*_edge_size)

    x0_f=tf.floor(x)
    y0_f=tf.floor(y)
    x1_f=x0_f+1

    x0=tf.cast(x0_f,tf.int32)
    y0=tf.cast(y0_f,tf.int32)
    x1=tf.cast(tf.minimum(x1_f,_width_f-1+2*_edge_size),tf.int32)

    dim2 = int((width + 2 * _edge_size))
    dim1 = int((width + 2 * _edge_size) * (height + 2 * _edge_size))

    # print('range batch',tf.range(num_batch))
    base=repeat(tf.range(num_batch)*dim1,height*width)
    base_y0 = base + y0 * dim2
    idx_l = base_y0 + x0
    idx_r = base_y0 + x1

    im_flat = tf.reshape(img, tf.stack([-1, num_channels]))

    pix_l = tf.gather(im_flat, idx_l)
    pix_r = tf.gather(im_flat, idx_r)

    weight_l = tf.expand_dims(x1_f - x, 1)
    weight_r = tf.expand_dims(x - x0_f, 1)

    return weight_l * pix_l + weight_r * pix_r

"""
Billinear sampler
"""
def billinear_sampler_fct(input_images, x_offset):
    # print("INPUT IMAGES SHAPE",input_images.shape)
    _num_batch=len(input_images)
    _height=1024
    _width=2048
    _num_channels=3
    # _height=input_images[0].shape[0]
    # _width=input_images[0].shape[1]
    # _num_channels=input_images[0].shape[2]


    _height_f=float(_height)
    _width_f=float(_width)

    """
        Transform
    """
    x_t, y_t = tf.meshgrid(tf.linspace(0.0,   _width_f - 1.0,  _width), tf.linspace(0.0 , _height_f - 1.0 , _height))

    x_t_flat = tf.reshape(x_t, (1, -1))
    y_t_flat = tf.reshape(y_t, (1, -1))

    x_t_flat = tf.tile(x_t_flat, tf.stack([_num_batch, 1]))
    y_t_flat = tf.tile(y_t_flat, tf.stack([_num_batch, 1]))

    x_t_flat = tf.reshape(x_t_flat, [-1])
    y_t_flat = tf.reshape(y_t_flat, [-1])
    
    offset_reshaped=tf.reshape(x_offset, [-1])
    offset_reshaped = tf.cast(offset_reshaped,dtype='float32')

    x_t_flat = x_t_flat +  offset_reshaped * _width_f

    input_transformed=interpolate(input_images,x_t_flat,y_t_flat,_num_batch,_height_f,_width_f,_num_channels)

    output = tf.reshape(input_transformed, tf.stack([_num_batch, _height, _width, _num_channels]))
    return output