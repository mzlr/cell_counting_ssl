import os
import argparse
from glob import glob
import tensorflow as tf
import numpy as np
import data
import model
import utils
from config import config as cfg
from config import (
    cfg_from_file, cfg_from_list, assert_and_infer_cfg, print_cfg)


def train(tf_config, logger):
    dataset = data.Dataset3D(cfg.DATASET, cfg.RNG_SEED, training=True)
    imgs, labels = dataset.preprocessing(
        augment=True, batch_size=cfg.TRAIN.BATCH_SIZE, num_epochs=cfg.TRAIN.EPOCH)

    dataset = data.Dataset3D('stn_unlabeled', cfg.RNG_SEED, training=True)
    imgs_aug, imgs_orig = dataset.preprocessing(
        augment=True, batch_size=cfg.TRAIN.BATCH_SIZE_AUG, num_epochs=cfg.TRAIN.EPOCH)
    if 'cutout' in cfg.TRAIN.SEMI_AUG:
        imgs_orig, mask = tf.split(imgs_orig, 2, axis=-1, num=2)
    elif 'mixup' in cfg.TRAIN.SEMI_AUG:
        factor = tf.stop_gradient(tf.random_uniform(
            [cfg.TRAIN.BATCH_SIZE_AUG//2, 1, 1, 1, 1]))
        tf.summary.scalar('mixup_factor', tf.reduce_mean(factor))
        imgs_aug = factor * imgs_orig[:cfg.TRAIN.BATCH_SIZE_AUG//2, :, :, :, :] + \
            (1-factor) * imgs_orig[cfg.TRAIN.BATCH_SIZE_AUG//2:, :, :, :, :]
    elif 'rot' in cfg.TRAIN.SEMI_AUG:
        rot_flag = tf.random_uniform((), minval=0, maxval=2, dtype=tf.int32)
        imgs_aug = tf.cond(
            tf.equal(rot_flag, 0),
            lambda: tf.transpose(tf.reverse(imgs_aug, [2]), [0, 2, 1, 3, 4]),
            lambda: tf.reverse(tf.transpose(imgs_aug, [0, 2, 1, 3, 4]), [2]),
            name='rot')

    net, _ = model.unet_3d(
        tf.concat([imgs, imgs_aug], axis=0), bn_training=True, layers=4,
        features_root=32, dropout_training=True, dataset=cfg.DATASET)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    if cfg.TRAIN.MULHEAD:
        density_sup = net[:cfg.TRAIN.BATCH_SIZE, :, :, :, :]
        density_aug = net[cfg.TRAIN.BATCH_SIZE:, :, :, :, :]
        with tf.variable_scope('cls'):
            density_sup = tf.layers.conv3d(
                density_sup, 1, 1, activation=tf.nn.relu)
        with tf.variable_scope('cls_noise'):
            density_aug = tf.layers.conv3d(
                density_aug, 1, 1, activation=tf.nn.relu)
    else:
        with tf.variable_scope('cls'):
            net = tf.layers.conv3d(net, 1, 1, activation=tf.nn.relu)
            density_sup = net[:cfg.TRAIN.BATCH_SIZE, :, :, :, :]
            density_aug = net[cfg.TRAIN.BATCH_SIZE:, :, :, :, :]
    if 'rot' in cfg.TRAIN.SEMI_AUG:
        density_aug = tf.cond(
            tf.equal(rot_flag, 0),
            lambda: tf.reverse(tf.transpose(
                density_aug, [0, 2, 1, 3, 4]), [2]),
            lambda: tf.transpose(tf.reverse(
                density_aug, [2]), [0, 2, 1, 3, 4]),
            name='rot_rev')

    net, _ = model.unet_3d(
        imgs_orig, bn_training=True, layers=4, features_root=32,
        dropout_training=False, dataset=cfg.DATASET, reuse=True)
    with tf.variable_scope('cls', reuse=True):
        density_orig = tf.layers.conv3d(net, 1, 1, activation=tf.nn.relu)
    if 'cutout' in cfg.TRAIN.SEMI_AUG:
        density_orig = density_orig * mask
    elif 'mixup' in cfg.TRAIN.SEMI_AUG:
        density_orig = factor * density_orig[:cfg.TRAIN.BATCH_SIZE_AUG//2, :, :, :, :] + \
            (1-factor) * density_orig[cfg.TRAIN.BATCH_SIZE_AUG//2:, :, :, :, :]

    if cfg.TRAIN.ANNEAL:
        weights = utils.anneal_sup_loss(
            density_sup, labels, tf.train.get_or_create_global_step())
    else:
        weights = 1.

    if cfg.TRAIN.LOSS == 'mse':
        loss_sup = tf.losses.mean_squared_error(
            labels=labels * cfg.MODEL.RATIO[cfg.DATASET], predictions=density_sup, weights=weights)
    elif cfg.TRAIN.LOSS == 'ssim':
        loss_sup = tf.losses.mean_squared_error(
            labels=labels * cfg.MODEL.RATIO[cfg.DATASET], predictions=density_sup, weights=weights)
        if cfg.TRAIN.SSIM_W:
            weights = tf.stop_gradient(
                tf.cast(tf.equal(labels, 1.), tf.float32))
            tf.summary.scalar('ssim_w_ratio', tf.reduce_mean(weights))
        else:
            weights = 1.
        ssim = utils.ssim(
            density_sup, tf.stop_gradient(
                labels * cfg.MODEL.RATIO[cfg.DATASET]),
            max_val=cfg.MODEL.RATIO[cfg.DATASET], weights=weights)
        tf.summary.scalar('loss_ssim', ssim)
        loss_sup += ssim
    else:
        raise ValueError('wrong loss')
    density_orig = tf.stop_gradient(density_orig)

    if cfg.TRAIN.PEAK:
        max_filter = tf.layers.max_pooling3d(
            density_orig, pool_size=3, strides=1, padding='same')
        weights = tf.cast(tf.equal(max_filter, density_orig), tf.float32)
        if cfg.TRAIN.KERNEL == 'gaussian':
            kernel = utils.gauss3D((5, 5, 5), 1.)[..., np.newaxis, np.newaxis]
            kernel = tf.constant(kernel, tf.float32)
            weights = tf.nn.conv3d(
                weights, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')
        elif cfg.TRAIN.KERNEL == 'maximum':
            weights = tf.layers.max_pooling3d(
                weights, pool_size=5, strides=1, padding='same')
        else:
            raise ValueError('wrong TRAIN.KERNEL')
        tf.summary.scalar('peak_ratio', tf.reduce_mean(weights))
    else:
        weights = 1.

    loss_aug = tf.losses.mean_squared_error(
        labels=density_orig, predictions=density_aug, weights=tf.stop_gradient(weights))

    if cfg.TRAIN.EMA == 'ema':
        ema = tf.train.ExponentialMovingAverage(decay=0.999)
        ema_op = ema.apply(tf.trainable_variables())
        update_ops.append(ema_op)

    if cfg.SOLVER.OPT == 'adam':
        lr_decayed = tf.train.cosine_decay_restarts(
            cfg.SOLVER.BASE_LR, tf.train.get_or_create_global_step(), cfg.SOLVER.RESTART_STEP)
        wd = cfg.SOLVER.WEIGHT_DECAY * lr_decayed / cfg.SOLVER.BASE_LR
        optimizer = tf.contrib.opt.AdamWOptimizer(wd, learning_rate=lr_decayed)
    elif cfg.SOLVER.OPT == 'momentum':
        global_step = tf.cast(tf.train.get_or_create_global_step(), tf.float32)
        lr_decayed = cfg.SOLVER.BASE_LR * \
            tf.pow((1. - global_step / cfg.TRAIN.STEP), 0.9)
        wd = cfg.SOLVER.WEIGHT_DECAY * lr_decayed / cfg.SOLVER.BASE_LR
        optimizer = tf.contrib.opt.MomentumWOptimizer(
            wd, learning_rate=lr_decayed, momentum=0.9, use_nesterov=True)

    if cfg.TRAIN.SEMI_SUP:
        if cfg.TRAIN.RAMP:
            global_step = tf.cast(
                tf.train.get_or_create_global_step(), tf.float32)
            ratio = tf.clip_by_value(global_step / 150, 0., 1.)
            w = tf.stop_gradient(tf.exp(5.0 * (ratio - 1.)) * cfg.TRAIN.RAMP_W)
            tf.summary.scalar('w_aug', w)
        else:
            w = cfg.TRAIN.RAMP_W
        step_pixel = optimizer.minimize(
            loss_sup + w * loss_aug, global_step=tf.train.get_or_create_global_step())
    else:
        step_pixel = optimizer.minimize(
            loss_sup, global_step=tf.train.get_or_create_global_step())

    tf.summary.scalar('loss_sup', loss_sup)
    tf.summary.scalar('loss_aug', loss_aug)
    tf.summary.image('img', imgs[:, :, :, 10, :], max_outputs=1)
    tf.summary.image('img_aug', imgs_aug[:, :, :, 10, :], max_outputs=1)
    tf.summary.image('img_orig', imgs_orig[:, :, :, 10, :], max_outputs=1)
    tf.summary.image('gt', labels[:, :, :, 10, :], max_outputs=1)
    tf.summary.image('pred', density_sup[:, :, :, 10, :], max_outputs=1)
    tf.summary.image('pred_aug', density_aug[:, :, :, 10, :], max_outputs=1)
    tf.summary.image('pred_orig', density_orig[:, :, :, 10, :], max_outputs=1)
    merged = tf.summary.merge_all()

    with tf.control_dependencies([step_pixel]):
        step = tf.group(*update_ops)

    saver = tf.train.Saver(max_to_keep=1000)
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.mkdir(cfg.OUTPUT_DIR)

    with tf.Session(config=tf_config) as sess:
        summary_writer = tf.summary.FileWriter(
            os.path.join(cfg.OUTPUT_DIR, 'train'), sess.graph)
        if tf.train.latest_checkpoint(cfg.OUTPUT_DIR) is None:
            start_step = 0
            sess.run(tf.global_variables_initializer())
            logger.info('Saving path is {}'.format(cfg.OUTPUT_DIR))
        else:
            weights_path = tf.train.latest_checkpoint(cfg.OUTPUT_DIR)
            start_step = int(weights_path.split('-')[-1])
            tf.train.Saver().restore(sess, weights_path)
            logger.info('Restoring weights from {}'.format(weights_path))
        logger.info('Training at Step {}'.format(start_step + 1))

        for i in range(start_step, cfg.TRAIN.STEP):
            if i % cfg.LOG_PERIOD == 0 or i == cfg.TRAIN.STEP - 1:
                loss_sup_val, loss_aug_val, summary, _ = sess.run(
                    [loss_sup, loss_aug, merged, step])
                summary_writer.add_summary(summary, i + 1)
                logger.info('Step:{}/{} loss_sup:{:6.3f} loss_aug:{:6.3f}'.format(
                    i + 1, cfg.TRAIN.STEP, loss_sup_val, loss_aug_val))
            else:
                sess.run([step])

            if i + 1 in [50, 150, 350, 750, 1550, 3150, 6350, 12750, 25550] or \
                    i == cfg.TRAIN.STEP - 1:
                weights_path = saver.save(
                    sess, os.path.join(cfg.OUTPUT_DIR, 'model'),
                    global_step=i + 1, write_meta_graph=False)
                logger.info('Saving weights to {}'.format(weights_path))
    tf.reset_default_graph()


def test(tf_config, logger):
    cfg.TRAIN.BATCH_SIZE = 1
    dataset = data.Dataset3D('stn_eval', cfg.RNG_SEED, training=False)
    imgs, _ = dataset.preprocessing(
        augment=False, batch_size=cfg.TRAIN.BATCH_SIZE, num_epochs=-1)

    net, _ = model.unet_3d(imgs, bn_training=False, layers=4, features_root=32,
                           dropout_training=False, dataset=cfg.DATASET)
    with tf.variable_scope('cls'):
        net = tf.layers.conv3d(net, 1, 1, activation=tf.nn.relu)

    if cfg.TRAIN.EMA == 'ema':
        ema = tf.train.ExponentialMovingAverage(decay=0.999)
        saver = tf.train.Saver(
            ema.variables_to_restore(tf.trainable_variables()))
    else:
        saver = tf.train.Saver()
    with tf.Session(config=tf_config) as sess:
        ckpts = [ckpt[:-6]
                 for ckpt in glob(os.path.join(cfg.OUTPUT_DIR, '*.index'))]
        for weights_path in ckpts:
            logger.info('Restoring weights from {}'.format(weights_path))
            saver.restore(sess, weights_path)

            step_num = (dataset.total_num -
                        dataset.train_num) // cfg.TRAIN.BATCH_SIZE

            pred_val_density = []
            for i in range(step_num):
                net_val = sess.run(net)

                pred_val_density.append(net_val)
                logger.info('#{}'.format(i))

            utils.save_results(
                os.path.join(cfg.OUTPUT_DIR,
                             'result_eval_step_{}.hdf5'.format(int(weights_path.split('-')[-1]))),
                {'density': pred_val_density})
    tf.reset_default_graph()


def main(_):
    parser = argparse.ArgumentParser(
        description='Classification model training')
    parser.add_argument('--config_file', type=str, default=None,
                        help='Optional config file for params')
    parser.add_argument('opts', help='see config.py for all options',
                        default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    if args.config_file is not None:
        cfg_from_file(args.config_file)
    if args.opts is not None:
        cfg_from_list(args.opts)

    assert_and_infer_cfg()
    print_cfg()

    # os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.GPU_ID)
    logger = utils.setup_custom_logger('root')
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    tf_config = tf.ConfigProto(device_count=dict(
        GPU=1), gpu_options=tf.GPUOptions(allow_growth=True))
    # effective_batch_size = cfg.TRAIN.BATCH_SIZE + \
    #     cfg.TRAIN.BATCH_SIZE_AUG if cfg.TRAIN.SEMI_SUP else cfg.TRAIN.BATCH_SIZE
    # if effective_batch_size <= 4:
    # tf_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    tf.enable_resource_variables()

    train(tf_config, logger)
    test(tf_config, logger)


if __name__ == '__main__':
    utils.set_flags()
    tf.app.run()
