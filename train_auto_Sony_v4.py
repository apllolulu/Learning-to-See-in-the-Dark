# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division
import os, time, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import rawpy
import glob
from util import SE_Block
from tflearn.layers.conv import global_avg_pool

"""
resnet + seblock + branch net + Sigmoid

"""

input_dir = './dataset/Sony/short/'
gt_dir = './dataset/Sony/long/'
checkpoint_dir = './result_auto_Sony_v4/'
result_dir = './result_auto_Sony_v4/'

# get train IDs
train_fns = glob.glob(gt_dir + '0*.ARW')
train_ids = [int(os.path.basename(train_fn)[0:5]) for train_fn in train_fns]
#print("train_ids:",len(train_ids))#161

ps = 512  # patch size for training
save_freq = 100

DEBUG = 0
if DEBUG == 1:
    save_freq = 2
    train_ids = train_ids[0:5]


def lrelu(x):
    return tf.maximum(x * 0.2, x)

# 上采样 反卷积 连接
def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output

# 新添加 resnet + se_block
def ResnetBlock(x, dim, ksize, rate=1, activation_fn=lrelu,scope='rb'):
    with tf.variable_scope(scope):
        net = slim.conv2d(x, dim, [ksize, ksize], rate=rate, activation_fn=activation_fn, scope='conv1')
        #net = slim.conv2d(net, dim, [ksize, ksize], rate=rate, activation_fn=activation_fn, scope='conv2')
        net = SE_Block.Squeeze_excitation_layer(net, dim, ratio=4, layer_name=scope)
        return net + x

def Fully_connected(x, units, layer_name='fully_connected') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=True, units=units)

def Relu(x):
    return tf.nn.relu(x)

def Sigmoid(x):
    return tf.nn.sigmoid(x)

def network(input):

    temp0 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_temp_conv1_0')
    temp1 = ResnetBlock(temp0, 32, 3, rate=1, activation_fn=lrelu, scope='g_temp_conv1_1')
    temp2 = ResnetBlock(temp1, 32, 3, rate=1, activation_fn=lrelu, scope='g_temp_conv1_2')

    temp3 = global_avg_pool(temp2, name='GlobalAvgPool')  # 32

    temp4 = Fully_connected(temp3, units=10, layer_name='fully_connected1')
    temp5 = Relu(temp4)
    temp6 = Fully_connected(temp5, units=1, layer_name= 'fully_connected2')
    #ratio = min(ratio,300)
    ratio = Sigmoid(temp6)*300


    conv1 = slim.conv2d(input*ratio, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_0')
    conv1 = ResnetBlock(conv1, 32, 3, rate=1, activation_fn=lrelu, scope='g_conv1_1')
    conv1 = ResnetBlock(conv1, 32, 3, rate=1, activation_fn=lrelu, scope='g_conv1_2')
    pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

    conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_0')
    conv2 = ResnetBlock(conv2, 64, 3, rate=1, activation_fn=lrelu, scope='g_conv2_1')
    conv2 = ResnetBlock(conv2, 64, 3, rate=1, activation_fn=lrelu, scope='g_conv2_2')
    pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

    conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_0')
    conv3 = ResnetBlock(conv3, 128, 3, rate=1, activation_fn=lrelu, scope='g_conv3_1')
    conv3 = ResnetBlock(conv3, 128, 3, rate=1, activation_fn=lrelu, scope='g_conv3_2')
    pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

    conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_0')
    conv4 = ResnetBlock(conv4, 256, 3, rate=1, activation_fn=lrelu, scope='g_conv4_1')
    conv4 = ResnetBlock(conv4, 256, 3, rate=1, activation_fn=lrelu, scope='g_conv4_2')
    pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

    conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_0')
    conv5 = ResnetBlock(conv5, 512, 3, rate=1, activation_fn=lrelu, scope='g_conv5_1')
    conv5 = ResnetBlock(conv5, 512, 3, rate=1, activation_fn=lrelu, scope='g_conv5_2')

    up6 = upsample_and_concat(conv5, conv4, 256, 512)

    conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_0')
    conv6 = ResnetBlock(conv6, 256, 3, rate=1, activation_fn=lrelu, scope='g_conv6_1')
    conv6 = ResnetBlock(conv6, 256, 3, rate=1, activation_fn=lrelu, scope='g_conv6_2')

    up7 = upsample_and_concat(conv6, conv3, 128, 256)

    conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_0')
    conv7 = ResnetBlock(conv7, 128, 3, rate=1, activation_fn=lrelu, scope='g_conv7_1')
    conv7 = ResnetBlock(conv7, 128, 3, rate=1, activation_fn=lrelu, scope='g_conv7_2')

    up8 = upsample_and_concat(conv7, conv2, 64, 128)

    conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_0')
    conv8 = ResnetBlock(conv8, 64, 3, rate=1, activation_fn=lrelu, scope='g_conv8_1')
    conv8 = ResnetBlock(conv8, 64, 3, rate=1, activation_fn=lrelu, scope='g_conv8_2')

    up9 = upsample_and_concat(conv8, conv1, 32, 64)

    conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_0')
    conv9 = ResnetBlock(conv9, 32, 3, rate=1, activation_fn=lrelu, scope='g_conv9_1')
    conv9 = ResnetBlock(conv9, 32, 3, rate=1, activation_fn=lrelu, scope='g_conv9_2')

    conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')

    out = tf.depth_to_space(conv10, 2)
    return out,ratio


def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    # 输入 图片变为原始的一半
    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


sess = tf.Session()
in_image = tf.placeholder(tf.float32, [None, None, None, 4])
gt_image = tf.placeholder(tf.float32, [None, None, None, 3])
ratio_gt = tf.placeholder(tf.float32)

# 4 通道  变 3 通道
out_image,pre_ratio = network(in_image)

G_loss = tf.reduce_mean(tf.abs(out_image - gt_image))
G_loss_ratio = tf.log(tf.abs(pre_ratio - ratio_gt))
G_total_loss = G_loss+tf.log(G_loss_ratio)

t_vars = tf.trainable_variables()
lr = tf.placeholder(tf.float32)
G_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_total_loss)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)


g_loss = np.zeros((5000, 1))

allfolders = glob.glob('./result_auto_Sony_v4/*0')
lastepoch = 0

for folder in allfolders:
    lastepoch = np.maximum(lastepoch, int(folder[-4:]))

learning_rate = 1e-4
# epoch 4000
for epoch in range(lastepoch, 4001):

    if os.path.isdir("result_auto_Sony_v4/%04d" % epoch):
        continue
    cnt = 0
    if epoch > 2000:
        learning_rate = 1e-5

    for ind in np.random.permutation(len(train_ids)):
        # get the path from image id
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
        in_exposure = float(in_fn[9:-5]) # 曝光时间
        gt_exposure = float(gt_fn[9:-5])
        # 手动输入曝光比例
        ratio = min(gt_exposure / in_exposure, 300)

        st = time.time()
        cnt += 1

        # 输入图片 和 ground truth
        if input_images[str(ratio)[0:3]][ind] is None:
            raw = rawpy.imread(in_path)
            # 输入图片乘以曝光比例  * ratio
            input_images[str(ratio)[0:3]][ind] = np.expand_dims(pack_raw(raw), axis=0)

            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_images[ind] = np.expand_dims(np.float32(im / 65535.0), axis=0)

        # crop
        H = input_images[str(ratio)[0:3]][ind].shape[1]
        W = input_images[str(ratio)[0:3]][ind].shape[2]

        # ps = 512
        xx = np.random.randint(0, W - ps)
        yy = np.random.randint(0, H - ps)
        # 随机裁剪
        input_patch = input_images[str(ratio)[0:3]][ind][:, yy:yy + ps, xx:xx + ps, :]
        gt_patch = gt_images[ind][:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]

        if np.random.randint(2, size=1)[0] == 1:  # random flip 随机翻转
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)

        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2)
            gt_patch = np.flip(gt_patch, axis=2)

        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            input_patch = np.transpose(input_patch, (0, 2, 1, 3))
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))

        input_patch = np.minimum(input_patch, 1.0)
        """
        G_loss = tf.reduce_mean(tf.abs(out_image - gt_image))
        G_loss_ratio = tf.reduce_mean(tf.abs(pre_ratio - ratio))
        G_total_loss = G_loss+G_loss_ratio
            
        """
        _, G_current,G_current_ratio,G_current_total, output = sess.run([G_opt, G_loss,G_loss_ratio,G_total_loss, out_image],
                                        feed_dict={in_image: input_patch, gt_image: gt_patch, lr: learning_rate,ratio_gt:ratio})
        output = np.minimum(np.maximum(output, 0), 1)
        g_loss[ind] = G_current_total
        print("epoch %d" % epoch)
        print(" %d Current Loss=%.3f " % ( cnt, np.mean(G_current)))
        print(" %d Ratio Loss=%.3f " % ( cnt, np.mean(G_current_ratio)))
        print(" %d Toal Loss=%.3f " % ( cnt, np.mean(g_loss[np.where(g_loss)])))


        if epoch % save_freq == 0:
            if not os.path.isdir(result_dir + '%04d' % epoch):
                os.makedirs(result_dir + '%04d' % epoch)

            temp = np.concatenate((gt_patch[0, :, :, :], output[0, :, :, :]), axis=1)
            scipy.misc.toimage(temp * 255, high=255, low=0, cmin=0, cmax=255).save(
                result_dir + '%04d/%05d_00_train_%d.jpg' % (epoch, train_id, ratio))
            saver.save(sess, checkpoint_dir + 'model.ckpt')


