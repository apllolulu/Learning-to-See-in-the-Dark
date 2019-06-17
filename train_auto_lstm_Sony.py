# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division
import os, time, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import rawpy
import glob
from util.BasicConvLSTMCell import *
from datetime import datetime
from tflearn.layers.conv import global_avg_pool
from random import shuffle
input_dir = './dataset/Sony/short/'
gt_dir = './dataset/Sony/long/'
checkpoint_dir = './result_auto_lstm_Sony/'
result_dir = './result_auto_lstm_Sony/'

# get train IDs
train_fns = glob.glob(gt_dir + '0*.ARW')
train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]
#print("train_ids:",len(train_ids))#161

ps = 512  # patch size for training
save_freq = 50
n_levels = 3
scale = 0.5
ratio = 100
batch_size = 1

DEBUG = 0
if DEBUG == 1:
    save_freq = 2
    train_ids = train_ids[0:5]


def lrelu(x):
    return tf.maximum(x * 0.2, x)


def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output

# 新添加 resnet
def ResnetBlock(x, dim, ksize, rate=1, activation_fn=lrelu,scope='rb',reuse=False):
    with tf.variable_scope(scope):
        net = slim.conv2d(x, dim, [ksize, ksize], rate=rate, activation_fn=activation_fn, scope='conv1',reuse=reuse)
        net = slim.conv2d(net, dim, [ksize, ksize], rate=rate, activation_fn=activation_fn, scope='conv2',reuse=reuse)
        return net + x
"""
def network(input):
    #conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
    #conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
    
    conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_0')
    conv1 = ResnetBlock(conv1, 32, 3, rate=1, activation_fn=lrelu, scope='g_conv1_1')
    conv1 = ResnetBlock(conv1, 32, 3, rate=1, activation_fn=lrelu, scope='g_conv1_2')
    pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')


    #conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
    #conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_2')

    conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_0')
    conv2 = ResnetBlock(conv2, 64, 3, rate=1, activation_fn=lrelu, scope='g_conv2_1')
    conv2 = ResnetBlock(conv2, 64, 3, rate=1, activation_fn=lrelu, scope='g_conv2_2')
    pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')


    #conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
    #conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2')

    conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_0')
    conv3 = ResnetBlock(conv3, 128, 3, rate=1, activation_fn=lrelu, scope='g_conv3_1')
    conv3 = ResnetBlock(conv3, 128, 3, rate=1, activation_fn=lrelu, scope='g_conv3_2')
    pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

    #conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
    #conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2')

    conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_0')
    conv4 = ResnetBlock(conv4, 256, 3, rate=1, activation_fn=lrelu, scope='g_conv4_1')
    conv4 = ResnetBlock(conv4, 256, 3, rate=1, activation_fn=lrelu, scope='g_conv4_2')
    pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

    #conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
    #conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2')

    conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_0')
    conv5 = ResnetBlock(conv5, 512, 3, rate=1, activation_fn=lrelu, scope='g_conv5_1')
    conv5 = ResnetBlock(conv5, 512, 3, rate=1, activation_fn=lrelu, scope='g_conv5_2')

    up6 = upsample_and_concat(conv5, conv4, 256, 512)

    #conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1')
    #conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')

    conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_0')
    conv6 = ResnetBlock(conv6, 256, 3, rate=1, activation_fn=lrelu, scope='g_conv6_1')
    conv6 = ResnetBlock(conv6, 256, 3, rate=1, activation_fn=lrelu, scope='g_conv6_2')

    up7 = upsample_and_concat(conv6, conv3, 128, 256)

    #conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1')
    #conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')

    conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_0')
    conv7 = ResnetBlock(conv7, 128, 3, rate=1, activation_fn=lrelu, scope='g_conv7_1')
    conv7 = ResnetBlock(conv7, 128, 3, rate=1, activation_fn=lrelu, scope='g_conv7_2')

    up8 = upsample_and_concat(conv7, conv2, 64, 128)
    #conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1')
    #conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2')

    conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_0')
    conv8 = ResnetBlock(conv8, 64, 3, rate=1, activation_fn=lrelu, scope='g_conv8_1')
    conv8 = ResnetBlock(conv8, 64, 3, rate=1, activation_fn=lrelu, scope='g_conv8_2')

    up9 = upsample_and_concat(conv8, conv1, 32, 64)
    #conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1')
    #conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2')

    conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_0')
    conv9 = ResnetBlock(conv9, 32, 3, rate=1, activation_fn=lrelu, scope='g_conv9_1')
    conv9 = ResnetBlock(conv9, 32, 3, rate=1, activation_fn=lrelu, scope='g_conv9_2')

    conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')

    out = tf.depth_to_space(conv10, 2)
    return out
"""
def Fully_connected(x, units, reuse=False,scope='g_temp_fc',layer_name='fully_connected') :
    #with tf.name_scope(scope+layer_name) :
    with tf.variable_scope(scope+layer_name, reuse=tf.AUTO_REUSE):
        return tf.layers.dense(inputs=x, use_bias=True, units=units)

def Relu(x):
    return tf.nn.relu(x)

def generator(inputs, reuse=False, scope='g_net'):#tf.AUTO_REUSE
    # 输入数据的数量 高度 宽度 通道数
    inputs = tf.convert_to_tensor(inputs)
    n, h, w, c = inputs.get_shape().as_list()

    with tf.variable_scope('LSTM'):
        # shape, filter_size, num_features 初始化细胞cell
        cell = BasicConvLSTMCell([h / 4, w / 4], [3, 3], 128)
        rnn_state = cell.zero_state(batch_size=batch_size, dtype=tf.float32)


    x_unwrap = []
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):#tf.AUTO_REUSE
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            activation_fn=tf.nn.relu, padding='SAME', normalizer_fn=None,
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                            biases_initializer=tf.constant_initializer(0.0)):

            scale = 0.5
            inputs = slim.conv2d(inputs, 3, [3, 3], rate=1, activation_fn=lrelu, scope='g_temp_conv0_0')
            inp_pred = inputs
            for i in range(n_levels):
                # 三轮循环
                scale = scale ** (n_levels - i - 1)
                hi = int(round(h * scale))
                wi = int(round(w * scale))

                inp_blur = tf.image.resize_images(inputs, [hi, wi], method=0)
                inp_pred = tf.stop_gradient(tf.image.resize_images(inp_pred, [hi, wi], method=0))
                inp_all = tf.concat([inp_blur, inp_pred], axis=3, name='inp')
                rnn_state = tf.image.resize_images(rnn_state, [hi // 4, wi // 4], method=0)

                # 训练曝光度
                temp0 = slim.conv2d(inp_all, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_temp_conv1_0')
                temp1 = ResnetBlock(temp0, 32, 3, rate=1, activation_fn=lrelu, scope='g_temp_conv1_1')
                temp2 = ResnetBlock(temp1, 32, 3, rate=1, activation_fn=lrelu, scope='g_temp_conv1_2')

                temp3 = global_avg_pool(temp2, name='GlobalAvgPool')

                temp4 = Fully_connected(temp3, units=10, scope='g_temp_fc_1', layer_name='fully_connected1')
                temp5 = Relu(temp4)
                ratio = Fully_connected(temp5, units=1, scope='g_temp_fc_2', layer_name='fully_connected2')


                conv1_1 = slim.conv2d(inp_all*ratio, 32, [5, 5], scope='enc1_1')
                conv1_2 = ResnetBlock(conv1_1, 32, 5, scope='enc1_2')
                conv1_3 = ResnetBlock(conv1_2, 32, 5, scope='enc1_3')
                #conv1_4 = ResnetBlock(conv1_3, 32, 5, scope='enc1_4',reuse=tf.AUTO_REUSE)

                conv2_1 = slim.conv2d(conv1_3, 64, [5, 5], stride=2, scope='enc2_1')
                conv2_2 = ResnetBlock(conv2_1, 64, 5, scope='enc2_2')
                conv2_3 = ResnetBlock(conv2_2, 64, 5, scope='enc2_3')
                #conv2_4 = ResnetBlock(conv2_3, 64, 5, scope='enc2_4',reuse=tf.AUTO_REUSE)

                conv3_1 = slim.conv2d(conv2_3, 128, [5, 5], stride=2, scope='enc3_1')
                conv3_2 = ResnetBlock(conv3_1, 128, 5, scope='enc3_2')
                conv3_3 = ResnetBlock(conv3_2, 128, 5, scope='enc3_3')
                #conv3_4 = ResnetBlock(conv3_3, 128, 5, scope='enc3_4',reuse=tf.AUTO_REUSE)

                #deconv3_4, rnn_state = cell(conv3_4, rnn_state)
                deconv3_3 = conv3_3

                # decoder 解码器
                deconv3_3 = ResnetBlock(deconv3_3, 128, 5, scope='dec3_3')
                deconv3_2 = ResnetBlock(deconv3_3, 128, 5, scope='dec3_2')
                deconv3_1 = ResnetBlock(deconv3_2, 128, 5, scope='dec3_1')

                deconv2_4 = slim.conv2d_transpose(deconv3_1, 64, [4, 4], stride=2, scope='dec2_4')
                cat2 = deconv2_4 + conv2_3#conv2_4
                deconv2_3 = ResnetBlock(cat2, 64, 5, scope='dec2_3')
                deconv2_2 = ResnetBlock(deconv2_3, 64, 5, scope='dec2_2')
                deconv2_1 = ResnetBlock(deconv2_2, 64, 5, scope='dec2_1')

                deconv1_4 = slim.conv2d_transpose(deconv2_1, 32, [4, 4], stride=2, scope='dec1_4')
                cat1 = deconv1_4 + conv1_3#conv1_4
                deconv1_3 = ResnetBlock(cat1, 32, 5, scope='dec1_3')
                deconv1_2 = ResnetBlock(deconv1_3, 32, 5, scope='dec1_2')
                deconv1_1 = ResnetBlock(deconv1_2, 32, 5, scope='dec1_1')
                inp_pred = slim.conv2d(deconv1_1, 12, [1, 1], rate=1, activation_fn=None, scope='dec1_0')

                inp_pred = tf.depth_to_space(inp_pred, 2)

                # num_feature*2 后  通道数分为两部分  一部分为细胞状态 一部分为隐藏状态
                if i >= 0:
                    x_unwrap.append(inp_pred)
                if i == 0:
                    tf.get_variable_scope().reuse_variables()

        return x_unwrap

"""
def generator(inputs, reuse=False, scope='g_net'):
    # 输入数据的数量 高度 宽度 通道数
    n, h, w, c = inputs.get_shape().as_list()

    with tf.variable_scope('LSTM'):
        # shape, filter_size, num_features 初始化细胞cell
        cell = BasicConvLSTMCell([h / 4, w / 4], [3, 3], 128)
        rnn_state = cell.zero_state(batch_size=1, dtype=tf.float32)


    x_unwrap = []
    with tf.variable_scope(scope, reuse=reuse):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            activation_fn=tf.nn.relu, padding='SAME', normalizer_fn=None,
                            weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                            biases_initializer=tf.constant_initializer(0.0)):

            scale = 0.5
            inputs = slim.conv2d(inputs, 3, [3, 3], rate=1, activation_fn=lrelu, scope='g_temp_conv0',reuse=tf.AUTO_REUSE)
            inp_pred = inputs
            for i in range(n_levels):
                # 三轮循环
                scale = scale ** (n_levels - i - 1)
                hi = int(round(h * scale))
                wi = int(round(w * scale))

                inp_blur = tf.image.resize_images(inputs, [hi, wi], method=0)
                inp_pred = tf.stop_gradient(tf.image.resize_images(inp_pred, [hi, wi], method=0))
                inp_all = tf.concat([inp_blur, inp_pred], axis=3, name='inp')
                rnn_state = tf.image.resize_images(rnn_state, [hi // 4, wi // 4], method=0)

                # 训练曝光度
                temp0 = slim.conv2d(inp_all, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_temp_conv1_0',reuse=tf.AUTO_REUSE)
                temp1 = ResnetBlock(temp0, 32, 3, rate=1, activation_fn=lrelu, scope='g_temp_conv1_1',reuse=tf.AUTO_REUSE)
                temp2 = ResnetBlock(temp1, 32, 3, rate=1, activation_fn=lrelu, scope='g_temp_conv1_2',reuse=tf.AUTO_REUSE)

                temp3 = global_avg_pool(temp2, name='GlobalAvgPool')

                temp4 = Fully_connected(temp3, units=10, reuse=tf.AUTO_REUSE,scope='g_temp_fc_1', layer_name='fully_connected1')
                temp5 = Relu(temp4)
                ratio = Fully_connected(temp5, units=1,  reuse=tf.AUTO_REUSE,scope='g_temp_fc_2', layer_name='fully_connected2')


                conv1_1 = slim.conv2d(inp_all*ratio, 32, [5, 5], scope='enc1_1',reuse=tf.AUTO_REUSE)
                conv1_2 = ResnetBlock(conv1_1, 32, 5, scope='enc1_2',reuse=tf.AUTO_REUSE)
                conv1_3 = ResnetBlock(conv1_2, 32, 5, scope='enc1_3',reuse=tf.AUTO_REUSE)
                #conv1_4 = ResnetBlock(conv1_3, 32, 5, scope='enc1_4',reuse=tf.AUTO_REUSE)

                conv2_1 = slim.conv2d(conv1_3, 64, [5, 5], stride=2, scope='enc2_1',reuse=tf.AUTO_REUSE)
                conv2_2 = ResnetBlock(conv2_1, 64, 5, scope='enc2_2',reuse=tf.AUTO_REUSE)
                conv2_3 = ResnetBlock(conv2_2, 64, 5, scope='enc2_3',reuse=tf.AUTO_REUSE)
                #conv2_4 = ResnetBlock(conv2_3, 64, 5, scope='enc2_4',reuse=tf.AUTO_REUSE)

                conv3_1 = slim.conv2d(conv2_3, 128, [5, 5], stride=2, scope='enc3_1',reuse=tf.AUTO_REUSE)
                conv3_2 = ResnetBlock(conv3_1, 128, 5, scope='enc3_2',reuse=tf.AUTO_REUSE)
                conv3_3 = ResnetBlock(conv3_2, 128, 5, scope='enc3_3',reuse=tf.AUTO_REUSE)
                #conv3_4 = ResnetBlock(conv3_3, 128, 5, scope='enc3_4',reuse=tf.AUTO_REUSE)

                #deconv3_4, rnn_state = cell(conv3_4, rnn_state)
                deconv3_3 = conv3_3

                # decoder 解码器
                deconv3_3 = ResnetBlock(deconv3_3, 128, 5, scope='dec3_3',reuse=tf.AUTO_REUSE)
                deconv3_2 = ResnetBlock(deconv3_3, 128, 5, scope='dec3_2',reuse=tf.AUTO_REUSE)
                deconv3_1 = ResnetBlock(deconv3_2, 128, 5, scope='dec3_1',reuse=tf.AUTO_REUSE)

                deconv2_4 = slim.conv2d_transpose(deconv3_1, 64, [4, 4], stride=2, scope='dec2_4',reuse=tf.AUTO_REUSE)
                cat2 = deconv2_4 + conv2_3#conv2_4
                deconv2_3 = ResnetBlock(cat2, 64, 5, scope='dec2_3',reuse=tf.AUTO_REUSE)
                deconv2_2 = ResnetBlock(deconv2_3, 64, 5, scope='dec2_2',reuse=tf.AUTO_REUSE)
                deconv2_1 = ResnetBlock(deconv2_2, 64, 5, scope='dec2_1',reuse=tf.AUTO_REUSE)

                deconv1_4 = slim.conv2d_transpose(deconv2_1, 32, [4, 4], stride=2, scope='dec1_4',reuse=tf.AUTO_REUSE)
                cat1 = deconv1_4 + conv1_3#conv1_4
                deconv1_3 = ResnetBlock(cat1, 32, 5, scope='dec1_3',reuse=tf.AUTO_REUSE)
                deconv1_2 = ResnetBlock(deconv1_3, 32, 5, scope='dec1_2',reuse=tf.AUTO_REUSE)
                deconv1_1 = ResnetBlock(deconv1_2, 32, 5, scope='dec1_1',reuse=tf.AUTO_REUSE)
                inp_pred = slim.conv2d(deconv1_1, 12, [1, 1], rate=1, activation_fn=None, scope='dec1_0',reuse=tf.AUTO_REUSE)

                inp_pred = tf.depth_to_space(inp_pred, 2)

                # num_feature*2 后  通道数分为两部分  一部分为细胞状态 一部分为隐藏状态
                if i >= 0:
                    x_unwrap.append(inp_pred)
                if i == 0:
                    tf.get_variable_scope().reuse_variables()

        return x_unwrap
"""
def im2uint8(x):
    if x.__class__ == tf.Tensor:
        return tf.cast(tf.clip_by_value(x, 0.0, 1.0) * 255.0, tf.uint8)
    else:
        t = np.clip(x, 0.0, 1.0) * 255.0
        return t.astype(np.uint8)

def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out

"""
def input_producer(train_batch_ids):
    for ind in np.random.permutation(len(train_batch_ids)):
        gt_images = [None] * 6000
        input_images = {}
        input_images['300'] = [None] * len(train_batch_ids)  # 161
        input_images['250'] = [None] * len(train_batch_ids)
        input_images['100'] = [None] * len(train_batch_ids)

        train_id = train_batch_ids[ind]
        in_files = glob.glob(input_dir + '%05d_00*.ARW' % train_id)
        in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
        in_fn = os.path.basename(in_path)

        gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % train_id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        in_exposure = float(in_fn[9:-5])  # 曝光时间
        gt_exposure = float(gt_fn[9:-5])
        # 手动输入曝光比例
        ratio = min(gt_exposure / in_exposure, 300)

        # 输入图片 和 ground truth
        if input_images[str(ratio)[0:3]][ind] is None:
            raw = rawpy.imread(in_path)
            input_images[str(ratio)[0:3]][ind] = np.expand_dims(pack_raw(raw), axis=0) * ratio
            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_images[ind] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        # crop
        H = input_images[str(ratio)[0:3]][ind].shape[1]
        W = input_images[str(ratio)[0:3]][ind].shape[2]

        xx = np.random.randint(0, W - ps)
        yy = np.random.randint(0, H - ps)
        input_patch = input_images[str(ratio)[0:3]][ind][:, yy:yy + ps, xx:xx + ps, :]
        gt_patch = gt_images[ind][:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]

        if np.random.randint(2, size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2)
            gt_patch = np.flip(gt_patch, axis=2)
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            input_patch = np.transpose(input_patch, (0, 2, 1, 3))
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

        input_patch = np.minimum(input_patch, 1.0)
        #print(input_patch.shape) # (1, 512, 512, 4)

            #input_patch_total.append(input_patch)
            #gt_patch_total.append(gt_patch)

            #if len(input_patch_total)>=4:
                #input_patch_total = tf.stack(input_patch_total, axis=0)
                #gt_patch_total = tf.stack(gt_patch_total, axis=0)
        yield input_patch,gt_patch
                #input_patch_total = []
                #gt_patch_total = []
"""

def chunker(seq,size):
    return (seq[pos:pos+size] for pos in range(0,len(seq),size))

"""

def data_generator(train_ids):
    
    for ind in np.random.permutation(len(train_ids)):
        gt_images = [None] * 6000
        input_images = {}
        input_images['300'] = [None] * len(train_ids)  # 161
        input_images['250'] = [None] * len(train_ids)
        input_images['100'] = [None] * len(train_ids)

        train_id = train_ids[ind]
        in_files = glob.glob(input_dir + '%05d_00*.ARW' % train_id)
        in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
        in_fn = os.path.basename(in_path)

        gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % train_id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        
        in_exposure = float(in_fn[9:-5])  # 曝光时间
        gt_exposure = float(gt_fn[9:-5])
        # 手动输入曝光比例
        ratio = min(gt_exposure / in_exposure, 300)

        # 输入图片 和 ground truth
        if input_images[str(ratio)[0:3]][ind] is None:
            raw = rawpy.imread(in_path)
            
            input_images[str(ratio)[0:3]][ind] = np.expand_dims(pack_raw(raw), axis=0) * ratio
            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_images[ind] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        # crop
        H = input_images[str(ratio)[0:3]][ind].shape[1]
        W = input_images[str(ratio)[0:3]][ind].shape[2]

        xx = np.random.randint(0, W - ps)
        yy = np.random.randint(0, H - ps)
        input_patch = input_images[str(ratio)[0:3]][ind][:, yy:yy + ps, xx:xx + ps, :]
        gt_patch = gt_images[ind][:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]

        if np.random.randint(2, size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2)
            gt_patch = np.flip(gt_patch, axis=2)
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            input_patch = np.transpose(input_patch, (0, 2, 1, 3))
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

        input_patch = np.minimum(input_patch, 1.0)
        #print("input_patch.shape:",input_patch.shape) # (1, 512, 512, 4)
        input_patch = tf.convert_to_tensor(input_patch)

        yield input_patch, gt_patch
"""


def data_generator():

    def get_batches_fn(batch_size):

        index = [ind for ind in np.random.permutation(len(train_ids))]
        shuffle(index)

        for batch_i in range(0, len(index), batch_size):
            images = []
            gt_images = []
            for ind in index[batch_i:batch_i + batch_size]:
                train_id = train_ids[ind]
                in_files = glob.glob(input_dir + '%05d_00*.ARW' % train_id)
                in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
                in_fn = os.path.basename(in_path)

                gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % train_id)
                gt_path = gt_files[0]
                gt_fn = os.path.basename(gt_path)

                in_exposure = float(in_fn[9:-5])  # 曝光时间
                gt_exposure = float(gt_fn[9:-5])
                # 手动输入曝光比例
                ratio = min(gt_exposure / in_exposure, 300)

                raw = rawpy.imread(in_path)
                raw = pack_raw(raw)* ratio

                gt_raw = rawpy.imread(gt_path)
                im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
                im = np.float32(im / 65535.0)

                # crop
                H = raw.shape[0]
                W = raw.shape[1]

                xx = np.random.randint(0, W - ps)
                yy = np.random.randint(0, H - ps)

                input_patch = raw[yy:yy + ps, xx:xx + ps, :]
                gt_patch = im[yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]

                if np.random.randint(2, size=1)[0] == 1:  # random flip
                    input_patch = np.flip(input_patch, axis=0)
                    gt_patch = np.flip(gt_patch, axis=0)
                if np.random.randint(2, size=1)[0] == 1:
                    input_patch = np.flip(input_patch, axis=1)
                    gt_patch = np.flip(gt_patch, axis=1)
                if np.random.randint(2, size=1)[0] == 1:  # random transpose
                    input_patch = np.transpose(input_patch, (1, 0, 2))
                    gt_patch = np.transpose(gt_patch, (1, 0, 2))

                input_patch = np.minimum(input_patch, 1.0)

                images.append(input_patch)
                gt_images.append(gt_patch)
            yield np.array(images), np.array(gt_images)

    return get_batches_fn



def build_model():

    get_batches_fn = data_generator()
    loss_total = 0
    for input_patch, gt_patch in get_batches_fn(batch_size):
        x_unwrap = generator(input_patch, reuse=False, scope='g_net')

        n_levels = 3
        for i in range(n_levels):
            # 逐层计算损失 MSE
            _, hi, wi, _ = x_unwrap[i].get_shape().as_list()
            gt_i = tf.image.resize_images(gt_patch, [hi, wi], method=0)
            loss = tf.reduce_mean((gt_i - x_unwrap[i]) ** 2)
            loss_total += loss

    # training vars
    all_vars = tf.trainable_variables()

    g_vars = [var for var in all_vars if 'g_net' in var.name]
    lstm_vars = [var for var in all_vars if 'LSTM' in var.name]
    for var in all_vars:
        print(var.name)
    return loss_total,all_vars

def train():
    def get_optimizer(loss,lr,global_step=None, var_list=None, is_gradient_clip=False):
        train_op = tf.train.AdamOptimizer(lr)
        if is_gradient_clip:
            grads_and_vars = train_op.compute_gradients(loss, var_list=var_list)
            unchanged_gvs = [(grad, var) for grad, var in grads_and_vars if not 'LSTM' in var.name]
            rnn_grad = [grad for grad, var in grads_and_vars if 'LSTM' in var.name]
            rnn_var = [var for grad, var in grads_and_vars if 'LSTM' in var.name]
            capped_grad, _ = tf.clip_by_global_norm(rnn_grad, clip_norm=3)
            capped_gvs = list(zip(capped_grad, rnn_var))
            train_op = train_op.apply_gradients(grads_and_vars=capped_gvs + unchanged_gvs, global_step=global_step)
        else:
            train_op = train_op.minimize(loss, global_step, var_list)
        return train_op

    global_step = tf.Variable(initial_value=0, dtype=tf.int32, trainable=False)

    loss_total,all_vars = build_model()

    learning_rate = 1e-4
    epoch = 4000
    data_size = len(train_ids)
    max_steps = int(epoch * data_size)

    lr = tf.train.polynomial_decay(learning_rate, global_step, max_steps, end_learning_rate=0.0,
                                        power=0.3)

    # training operators
    train_gnet = get_optimizer(loss_total, lr , global_step, all_vars)

    # session and thread
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=25, keep_checkpoint_every_n_hours=1)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in range(sess.run(global_step), max_steps + 1):

        start_time = time.time()

        _, loss_total_val = sess.run([train_gnet, loss_total])

        duration = time.time() - start_time

        assert not np.isnan(loss_total_val), 'Model diverged with loss = NaN'

        if step % 5 == 0:
            num_examples_per_step = 1
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)

            format_str = ('%s: step %d, loss = (%.5f; %.5f, %.5f)(%.1f data/s; %.3f s/bch)')
            print(format_str % (datetime.now().strftime('%Y-%m-%d %H:%M:%S'), step, loss_total_val, 0.0,
                                0.0, examples_per_sec, sec_per_batch))


        if step % 1000 == 0 or step == max_steps:
            checkpoint_path = os.path.join(checkpoint_dir, 'checkpoints')

            model_name = "auto_lstm.model"
            saver.save(sess, os.path.join(checkpoint_path, model_name), global_step=step)

if __name__ == '__main__':
    train()


