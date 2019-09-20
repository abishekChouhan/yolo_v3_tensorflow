import numpy as np
import cv2
import tensorflow as tf


def upsample_layer(x, size_factor=2):
    # TODO: Do we need to set `align_corners` as True?
    if not isinstance(size_factor, int):
        raise ValueError
    new_height = x.get_shape()[1].value*size_factor
    new_width = x.get_shape()[2].value*size_factor
    x = tf.image.resize_nearest_neighbor(x, (new_height, new_width), name='upsampled_x'+str(size_factor))
    return x


def fixed_padding(inputs, kernel_size):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]], mode='CONSTANT')
    return padded_inputs


def darknet_conv2d(x, filters, kernel_size, stride=1, name='conv2d'):
    x_output_num = x.get_shape()[-1].value
    kernel_h, kernel_w = kernel_size
    if stride > 1:
        x = fixed_padding(x, kernel_size[0])
    pad = 'VALID' if stride == 2 else 'SAME'
    with tf.variable_scope(name):
        kernel = tf.get_variable(name+'kernel', [kernel_h, kernel_w, x_output_num, filters], tf.float64,
                                 tf.truncated_normal_initializer(stddev=5e-2, dtype=tf.float32))
        x = tf.nn.conv2d(x, kernel, strides=[1, stride, stride, 1], padding=pad, name=name)
        return x


def darknet_conv2d_bn_relu(x, filters, kernel_size=(1, 1), stride=1, name='conv2d_bn_relu'):
    x = darknet_conv2d(x, filters, kernel_size, stride, name=name)
    batch_size, x_output_num = x.get_shape()[0].value, x.get_shape()[-1].value
    epsilon = 0.000001
    with tf.variable_scope(name):
        mean, var = tf.nn.moments(x, [1, 2])
        mean, var = tf.reshape(mean, [batch_size, 1, 1, x_output_num]), tf.reshape(var, [batch_size, 1, 1, x_output_num])
        scale = tf.Variable(tf.ones([batch_size, 1, 1, x_output_num], dtype=tf.float64), dtype=tf.float64)
        beta = tf.Variable(tf.zeros([batch_size, 1, 1, x_output_num], dtype=tf.float64), dtype=tf.float64)
        x = tf.nn.batch_normalization(x, mean, var, beta, scale, epsilon)
        x = tf.nn.relu(x)
        return x


def residual_blocks(x, filters, num_blocks, name='res_body_'):
    ##add zero padding
    x = darknet_conv2d_bn_relu(x, filters, kernel_size=(3, 3), stride=2, name=name+'conv2d_1')
    x_shortcut = x
    name = name + 'resblock_'
    for i in range(num_blocks):
        name = name + str(i+1)
        x = darknet_conv2d_bn_relu(x, filters//2, kernel_size=(1, 1), name=name+'conv2d_1')
        x = darknet_conv2d_bn_relu(x, filters, kernel_size=(3, 3), name=name+'conv2d_2')
        x = x + x_shortcut
    return x


def darknet_body(x):
    x = darknet_conv2d_bn_relu(x, 32, (3, 3), name='conv2d_1')
    x = residual_blocks(x, 64, 1, name='res_body_1')
    x = residual_blocks(x, 128, 2, name='res_body_2')
    x = residual_blocks(x, 256, 8, name='res_body_3')
    route_1 = x
    x = residual_blocks(x, 512, 8, name='res_body_4')
    route_2 = x
    x = residual_blocks(x, 1024, 4, name='res_body_5')
    return route_1, route_2, x


def detection_layers(x, filters, out_filters, name='detection_block_'):
    x = darknet_conv2d_bn_relu(x, filters, kernel_size=(1, 1), name=name+'conv2d_1')
    x = darknet_conv2d_bn_relu(x, filters*2, kernel_size=(3, 3), name=name+'conv2d_2')
    x = darknet_conv2d_bn_relu(x, filters, kernel_size=(1, 1), name=name+'conv2d_3')
    x = darknet_conv2d_bn_relu(x, filters*2, kernel_size=(3, 3), name=name+'conv2d_4')
    x = darknet_conv2d_bn_relu(x, filters, kernel_size=(1, 1), name=name+'conv2d_5')
    route = x
    x = darknet_conv2d_bn_relu(x, filters*2, kernel_size=(3, 3), name=name+'conv2d_6')
    x = darknet_conv2d(x, out_filters, kernel_size=(1, 1), name=name+'conv2d_7')
    return route, x


def yolo_body(inputs, num_anchors, num_classes):
    output_features = num_anchors*(num_classes+5)
    route_1, route_2, route_3 = darknet_body(inputs)

    x, y1 = detection_layers(route_3, 512, output_features, name='detection_block1_')

    x = darknet_conv2d_bn_relu(x, 256, kernel_size=(1, 1), name='out_conv2d_1')
    x = upsample_layer(x, size_factor=2)

    x = tf.concat([x, route_2], axis=3)

    x, y2 = detection_layers(x, 256, output_features, name='detection_block2_')

    x = darknet_conv2d_bn_relu(x, 128, kernel_size=(1, 1), name='out_conv2d_2')
    x = upsample_layer(x, size_factor=2)

    x = tf.concat([x, route_1], axis=3)

    x, y3 = detection_layers(x, 128, output_features, name='detection_block3_')

    return y1, y2, y3


img = cv2.imread('img.jpg')
img = cv2.resize(img, (416, 416))
img = np.array([img], dtype='float64')
img = tf.constant(img, dtype=tf.float64)

y = yolo_body(img, 9, 2)

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    # sess.run(weights)
    y = sess.run(y)
    y1 = np.array(y[0])
    print(y1.shape)
    # cv2.imshow('img_file', y[0,:,:,1])
    # print(y.shape)
    # cv2.waitKey(0)
