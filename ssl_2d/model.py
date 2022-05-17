import logging
from collections import OrderedDict
import tensorflow as tf
from config import config as cfg
logger = logging.getLogger('root')


def conv2d(x, features, kernel_size, training, name, dataset):
    conv_2d = tf.layers.conv2d(x, features, kernel_size,
                               use_bias=True, activation=tf.nn.relu, padding='same', name=name)
    if cfg.MODEL.NORM == 'batch':
        conv_2d = tf.layers.batch_normalization(
            conv_2d, training=training, renorm=False,
            momentum=cfg.MODEL.BN_MOMENTUM, name=name + '_' + dataset)
    elif cfg.MODEL.NORM == 'instance':
        conv_2d = tf.contrib.layers.instance_norm(
            conv_2d, scope=name + '_' + dataset)
    return conv_2d


def deconv2d(x, features, kernel_size, stride, training, name, dataset):
    deconv_2d = tf.layers.conv2d_transpose(x, features, kernel_size, [stride, stride],
                                           use_bias=True, activation=tf.nn.relu, name=name)
    if cfg.MODEL.NORM == 'batch':
        deconv_2d = tf.layers.batch_normalization(
            deconv_2d, training=training, renorm=False,
            momentum=cfg.MODEL.BN_MOMENTUM, name=name + '_' + dataset)
    elif cfg.MODEL.NORM == 'instance':
        deconv_2d = tf.contrib.layers.instance_norm(
            deconv_2d, scope=name + '_' + dataset)
    return deconv_2d


def self_attn(x, training, name='self_attn'):
    with tf.variable_scope(name):
        batch_num, height, width, _ = tf.unstack(
            tf.shape(x), num=4)
        channel = x.get_shape().as_list()[-1]

        theta = tf.layers.conv2d(
            x, channel, 1, use_bias=True, activation=None)
        phi = tf.layers.conv2d(x, channel, 1,
                               use_bias=True, activation=None)
        g = tf.layers.conv2d(x, channel, 1,
                             use_bias=True, activation=None)

        theta = tf.reshape(theta, [batch_num, -1, channel])
        phi = tf.reshape(phi, [batch_num, -1, channel])
        phi = tf.transpose(phi, perm=[0, 2, 1])
        g = tf.reshape(g, [batch_num, -1, channel])

        theta_phi = tf.nn.softmax(tf.matmul(theta, phi), axis=2)

        if training:
            if cfg.MODEL.DROPOUT == 'aug':
                tmp = tf.layers.dropout(
                    theta_phi[cfg.TRAIN.BATCH_SIZE:, :, :],
                    training=training, rate=cfg.MODEL.DROPOUT_RATE)
                theta_phi = tf.concat(
                    [theta_phi[:cfg.TRAIN.BATCH_SIZE, :, :], tmp], axis=0)
            elif cfg.MODEL.DROPOUT == 'sup':
                tmp = tf.layers.dropout(
                    theta_phi[:cfg.TRAIN.BATCH_SIZE, :, :],
                    training=training, rate=cfg.MODEL.DROPOUT_RATE)
                theta_phi = tf.concat(
                    [tmp, theta_phi[cfg.TRAIN.BATCH_SIZE:, :, :]], axis=0)
            elif cfg.MODEL.DROPOUT == 'both':
                theta_phi = tf.layers.dropout(
                    theta_phi, training=training, rate=cfg.MODEL.DROPOUT_RATE)

        t = tf.matmul(theta_phi, g)
        t = tf.reshape(t, [batch_num, height, width, channel])
        t = tf.layers.conv2d(t, channel, 1, use_bias=False,
                             activation=tf.nn.relu)
        if cfg.MODEL.NORM == 'batch':
            t = tf.layers.batch_normalization(
                t, training=training, momentum=cfg.MODEL.BN_MOMENTUM)
        elif cfg.MODEL.NORM == 'instance':
            t = tf.contrib.layers.instance_norm(t)
        return t + x, theta_phi


def unet(in_node, bn_training=True, dropout_training=True, layers=4, features_root=32,
         filter_size=3, pool_size=2, scope='unet', reuse=False, dataset=''):
    """
    Creates a new convolutional unet for the given parametrization.
    :param in_node: input tensor, shape [?,nx,ny,channels]
    :param bn_training: whether BN training
    :param dropout_training: whether dropout training
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param scope: model scope
    """

    logger.info(
        "Layers:{layers}, features:{features}, filter size:{filter_size}x{filter_size}, "
        "pool size:{pool_size}x{pool_size}".format(
            layers=layers,
            features=features_root,
            filter_size=filter_size,
            pool_size=pool_size))

    pools = OrderedDict()
    dw_h_convs = OrderedDict()

    with tf.variable_scope(name_or_scope=scope, reuse=reuse):
        # down layers
        for layer in range(layers):
            features = 2 ** layer * features_root
            if layer == layers - 1:
                conv1 = conv2d(in_node, features, filter_size, bn_training,
                               "down_conv_{}_0_{}".format(str(layer), dataset), dataset)
                conv1 = conv2d(conv1, features, filter_size, bn_training,
                               "down_conv_{}_1_{}".format(str(layer), dataset), dataset)
                if cfg.SELF_ATTN == 1:
                    dw_h_convs[layer], theta_phi = self_attn(
                        conv1, dropout_training)
                elif cfg.SELF_ATTN == 0:
                    dw_h_convs[layer] = conv1
                    theta_phi = None
                else:
                    raise ValueError('incorrect SELF_ATTN')
            else:
                conv1 = conv2d(in_node, features, filter_size, bn_training,
                               "down_conv_{}_0".format(str(layer)), dataset)
                dw_h_convs[layer] = conv2d(conv1, features, filter_size, bn_training,
                                           "down_conv_{}_1".format(str(layer)), dataset)

            if layer < layers - 1:
                pools[layer] = tf.layers.max_pooling2d(
                    dw_h_convs[layer], pool_size=pool_size, strides=2, padding='valid')
                in_node = pools[layer]
        
        assert len(dw_h_convs) == layers
        if bn_training and dropout_training and 'layer_noise' in cfg.TRAIN.SEMI_AUG and not reuse:
            logger.info('layer noise used')
            for layer in range(layers):
                _, height, width, channel = tf.unstack(
                    tf.shape(dw_h_convs[layer]), num=4)
                noise = tf.random.uniform(
                    [1, height, width, channel], minval=-1, maxval=1) + 1.
                tmp = tf.stop_gradient(noise) * \
                    dw_h_convs[layer][cfg.TRAIN.BATCH_SIZE:, :, :, :]
                dw_h_convs[layer] = tf.concat(
                    [dw_h_convs[layer][:cfg.TRAIN.BATCH_SIZE, :, :, :], tmp], axis=0)
        in_node = dw_h_convs[layers - 1]

        # up layers
        for layer in range(layers - 2, -1, -1):
            features = 2 ** layer * features_root

            h_deconv = deconv2d(in_node, features, pool_size, pool_size, bn_training,
                                "deconv_{}".format(str(layer)), dataset)
            h_deconv_concat = tf.concat([dw_h_convs[layer], h_deconv], axis=3)

            conv1 = conv2d(h_deconv_concat, features, filter_size, bn_training,
                           "up_conv_{}_0".format(str(layer)), dataset)
            in_node = conv2d(conv1, features, filter_size, bn_training,
                             "up_conv_{}_1".format(str(layer)), dataset)

    return in_node, theta_phi


def conv3d(x, features, kernel_size, training, name, dataset):
    conv_3d = tf.layers.conv3d(x, features, kernel_size,
                               use_bias=True, activation=tf.nn.relu, padding='same', name=name)
    if cfg.MODEL.NORM == 'batch':
        conv_3d = tf.layers.batch_normalization(
            conv_3d, training=training, renorm=False,
            momentum=cfg.MODEL.BN_MOMENTUM, name=name + '_' + dataset)
    elif cfg.MODEL.NORM == 'instance':
        conv_3d = tf.contrib.layers.instance_norm(
            conv_3d, scope=name + '_' + dataset)
    return conv_3d


def deconv3d(x, features, kernel_size, stride, training, name, dataset):
    deconv_3d = tf.layers.conv3d_transpose(x, features, kernel_size, [stride, stride, stride],
                                           use_bias=True, activation=tf.nn.relu, name=name)
    if cfg.MODEL.NORM == 'batch':
        deconv_3d = tf.layers.batch_normalization(
            deconv_3d, training=training, renorm=False,
            momentum=cfg.MODEL.BN_MOMENTUM, name=name + '_' + dataset)
    elif cfg.MODEL.NORM == 'instance':
        deconv_3d = tf.contrib.layers.instance_norm(
            deconv_3d, scope=name + '_' + dataset)
    return deconv_3d


def self_attn3d(x, training, name='self_attn'):
    with tf.variable_scope(name):
        batch_num, height, width, depth, _ = tf.unstack(
            tf.shape(x), num=5)
        channel = x.get_shape().as_list()[-1]

        theta = tf.layers.conv3d(
            x, channel, 1, use_bias=True, activation=None)
        phi = tf.layers.conv3d(x, channel, 1,
                               use_bias=True, activation=None)
        g = tf.layers.conv3d(x, channel, 1,
                             use_bias=True, activation=None)

        theta = tf.reshape(theta, [batch_num, -1, channel])
        phi = tf.reshape(phi, [batch_num, -1, channel])
        phi = tf.transpose(phi, perm=[0, 2, 1])
        g = tf.reshape(g, [batch_num, -1, channel])

        theta_phi = tf.nn.softmax(tf.matmul(theta, phi), axis=2)

        if training:
            if cfg.MODEL.DROPOUT == 'aug':
                tmp = tf.layers.dropout(
                    theta_phi[cfg.TRAIN.BATCH_SIZE:, :, :],
                    training=training, rate=cfg.MODEL.DROPOUT_RATE)
                theta_phi = tf.concat(
                    [theta_phi[:cfg.TRAIN.BATCH_SIZE, :, :], tmp], axis=0)
            elif cfg.MODEL.DROPOUT == 'sup':
                tmp = tf.layers.dropout(
                    theta_phi[:cfg.TRAIN.BATCH_SIZE, :, :],
                    training=training, rate=cfg.MODEL.DROPOUT_RATE)
                theta_phi = tf.concat(
                    [tmp, theta_phi[cfg.TRAIN.BATCH_SIZE:, :, :]], axis=0)
            elif cfg.MODEL.DROPOUT == 'both':
                theta_phi = tf.layers.dropout(
                    theta_phi, training=training, rate=cfg.MODEL.DROPOUT_RATE)

        t = tf.matmul(theta_phi, g)
        t = tf.reshape(t, [batch_num, height, width, depth, channel])
        t = tf.layers.conv3d(t, channel, 1, use_bias=True,
                             activation=tf.nn.relu)
        if cfg.MODEL.NORM == 'batch':
            t = tf.layers.batch_normalization(
                t, training=training, momentum=cfg.MODEL.BN_MOMENTUM)
        elif cfg.MODEL.NORM == 'instance':
            t = tf.contrib.layers.instance_norm(t)
        return t + x, theta_phi


def unet_3d(in_node, bn_training=True, dropout_training=True, layers=4, features_root=32,
            filter_size=3, pool_size=2, scope='unet_3d', reuse=False, dataset=''):
    """
    Creates a new convolutional unet for the given parametrization.
    :param in_node: input tensor, shape [?,nx,ny,channels]
    :param bn_training: whether BN training
    :param dropout_training: whether dropout training
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param scope: model scope
    """

    logger.info(
        "Layers:{layers}, features:{features}, self-attention:{self_attn}, "
        "filter size:{filter_size}x{filter_size}x{filter_size}, "
        "pool size:{pool_size}x{pool_size}x{pool_size}".format(
            layers=layers,
            features=features_root,
            self_attn=cfg.SELF_ATTN,
            filter_size=filter_size,
            pool_size=pool_size))

    pools = OrderedDict()
    dw_h_convs = OrderedDict()

    with tf.variable_scope(name_or_scope=scope, reuse=reuse):
        # down layers
        for layer in range(layers):
            features = 2 ** layer * features_root
            if layer == layers - 1:
                # conv1 = conv3d(in_node, features, filter_size, bn_training,
                #                "down_conv_{}_0_{}".format(str(layer), dataset), dataset)
                # conv1 = conv3d(conv1, features, filter_size, bn_training,
                #                "down_conv_{}_1_{}".format(str(layer), dataset), dataset)
                # if cfg.SELF_ATTN == 1:
                #     dw_h_convs[layer], theta_phi = self_attn3d(
                #         conv1, dropout_training)
                # elif cfg.SELF_ATTN == 0:
                #     dw_h_convs[layer] = conv1
                #     theta_phi = None
                # else:
                #     raise ValueError('incorrect SELF_ATTN')
                if cfg.SELF_ATTN == 1:
                    conv1 = conv3d(in_node, features, filter_size, bn_training,
                                   "down_conv_{}_0_{}".format(str(layer), dataset), dataset)
                    conv1 = conv3d(conv1, features, filter_size, bn_training,
                                   "down_conv_{}_1_{}".format(str(layer), dataset), dataset)
                    dw_h_convs[layer], theta_phi = self_attn3d(
                        conv1, dropout_training)
                elif cfg.SELF_ATTN == 0:
                    conv1 = conv3d(in_node, features, filter_size, bn_training,
                                   "down_conv_{}_0".format(str(layer)), dataset)
                    conv1 = conv3d(conv1, features, filter_size, bn_training,
                                   "down_conv_{}_1".format(str(layer)), dataset)
                    dw_h_convs[layer] = conv1
                    theta_phi = None
                else:
                    raise ValueError('incorrect SELF_ATTN')
            else:
                conv1 = conv3d(in_node, features, filter_size, bn_training,
                               "down_conv_{}_0".format(str(layer)), dataset)
                dw_h_convs[layer] = conv3d(conv1, features, filter_size, bn_training,
                                           "down_conv_{}_1".format(str(layer)), dataset)

            if layer < layers - 1:
                pools[layer] = tf.layers.max_pooling3d(
                    dw_h_convs[layer], pool_size=pool_size, strides=2, padding='valid')
                in_node = pools[layer]

        assert len(dw_h_convs) == layers
        if bn_training and dropout_training and 'layer_noise' in cfg.TRAIN.SEMI_AUG and not reuse:
            logger.info('layer noise used')
            for layer in range(layers):
                _, height, width, depth, channel = tf.unstack(
                    tf.shape(dw_h_convs[layer]), num=5)
                noise = tf.random.uniform(
                    [1, height, width, depth, channel], minval=-1, maxval=1) + 1.
                tmp = tf.stop_gradient(noise) * \
                    dw_h_convs[layer][cfg.TRAIN.BATCH_SIZE:, :, :, :, :]
                dw_h_convs[layer] = tf.concat(
                    [dw_h_convs[layer][:cfg.TRAIN.BATCH_SIZE, :, :, :, :], tmp], axis=0)

        in_node = dw_h_convs[layers - 1]

        # up layers
        for layer in range(layers - 2, -1, -1):
            features = 2 ** layer * features_root

            h_deconv = deconv3d(in_node, features, pool_size, pool_size, bn_training,
                                "deconv_{}".format(str(layer)), dataset)
            h_deconv_concat = tf.concat([dw_h_convs[layer], h_deconv], axis=4)

            # if dropout_training and layer == 0:
            #     logger.info('dropout in the last upsample layer')
            #     if cfg.MODEL.DROPOUT == 'aug':
            #         net_aug = tf.layers.dropout(
            #             h_deconv_concat[cfg.TRAIN.BATCH_SIZE:, :, :, :, :],
            #             training=dropout_training,
            #             noise_shape=[cfg.TRAIN.BATCH_SIZE_AUG, 1, 1, 1, 64],
            #             rate=0.7)
            #         h_deconv_concat = tf.concat(
            #             [h_deconv_concat[:cfg.TRAIN.BATCH_SIZE, :, :, :, :], net_aug], axis=0)

            conv1 = conv3d(h_deconv_concat, features, filter_size, bn_training,
                           "up_conv_{}_0".format(str(layer)), dataset)
            in_node = conv3d(conv1, features, filter_size, bn_training,
                             "up_conv_{}_1".format(str(layer)), dataset)

            # if dropout_training and layer == 0:
            #     logger.info('dropout in the last upsample layer')
            #     if cfg.MODEL.DROPOUT == 'aug':
            #         net_aug = tf.layers.dropout(
            #             in_node[cfg.TRAIN.BATCH_SIZE:, :, :, :, :],
            #             training=dropout_training,
            #             noise_shape=[cfg.TRAIN.BATCH_SIZE_AUG, 1, 1, 1, 32],
            #             rate=cfg.MODEL.DROPOUT_RATE)
            #         in_node = tf.concat(
            #             [in_node[:cfg.TRAIN.BATCH_SIZE, :, :, :, :], net_aug], axis=0)

    return in_node, theta_phi
