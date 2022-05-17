import h5py
import os
import glob
import logging
import numpy as np
import tensorflow as tf
import utils
from config import config as cfg
logger = logging.getLogger('root')


data_path = {
    'vgg': 'data/VGG/VGG.hdf5',
    'dcc': 'data/DCC/DCC.hdf5',
    'mbm': 'data/MBM/MBM.hdf5',
    'adi': 'data/ADI/ADI.hdf5',
    'mbc': 'data/MBC/tfrecord',
    'stn_75': 'data/STN/075.hdf5',
    'stn_121': 'data/STN/121.hdf5',
    'stn_eval': 'data/STN/eval_set.hdf5',
    'stn_unlabeled': 'data/STN/unlabeled.hdf5', }

image_shape = {
    'vgg': (256, 256, 3),
    'dcc': (256, 256, 3),
    'mbm': (600, 600, 3),
    'adi': (152, 152, 3),
    'mbc': (512, 512),
    'stn_75': (224, 224, 64),
    'stn_121': (224, 224, 64),
    'stn_eval': (224, 224, 64),
    'stn_unlabeled': (256, 256, 64), }

train_num = {
    'vgg': 64,
    'dcc': 100,
    'mbm': 15,
    'adi': 50,
    'mbc': 58,
    'stn_75': 4,
    'stn_121': 4,
    'stn_eval': 0,
    'stn_unlabeled': 55, }

total_num = {
    'vgg': 200,
    'dcc': 176,
    'mbm': 44,
    'adi': 200,
    'mbc': 158,
    'stn_75': 16,
    'stn_121': 16,
    'stn_eval': 5,
    'stn_unlabeled': 55, }


class Dataset3D(object):
    def __init__(self, dataset, seed, training):
        self.dataset = dataset
        self.image_shape = image_shape[dataset]
        self.seed = seed
        self.total_num = total_num[self.dataset]
        self.train_num = cfg.TRAIN.NUM if self.dataset == 'stn_75' else train_num[
            self.dataset]
        self.training = training
        if self.dataset == 'stn_75' and cfg.TRAIN.LABEL_DISTANCE:
            data_path[dataset] = data_path[dataset].replace(
                '/075.hdf5', '/075_fid.hdf5')
        logger.info('Loading data from {}'.format(data_path[dataset]))
        np.random.seed(self.seed)
        ind = np.random.permutation(self.total_num)

        if self.dataset == 'mbc':
            record_files = glob.glob(os.path.join(data_path[dataset], '*'))
            assert len(record_files) == self.total_num
            ind = ind[:self.train_num] if self.training else ind[self.train_num:]
            record_files = [record_files[i] for i in ind]

            if self.training:
                assert len(record_files) == self.train_num
                self.data = tf.data.TFRecordDataset(
                    record_files,
                    buffer_size=100 * 1024 * 1024,  # 100 MiB per file
                    num_parallel_reads=tf.data.experimental.AUTOTUNE)
            else:
                assert len(record_files) == self.total_num - self.train_num
                self.data = tf.data.TFRecordDataset(record_files)
        elif self.dataset in ['stn_75', 'stn_121', 'stn_eval', 'stn_unlabeled']:
            with h5py.File(data_path[dataset], 'r') as hf:
                imgs = hf.get('imgs')[()]
                if self.dataset != 'stn_unlabeled':
                    counts = hf.get('counts')[()]
                    assert imgs.shape == counts.shape
            assert imgs.shape == (self.total_num,) + self.image_shape

            # rescale to [0, 1]
            imgs = (imgs - np.min(
                imgs, axis=(1, 2, 3), keepdims=True)) / (np.max(
                    imgs, axis=(1, 2, 3), keepdims=True) - np.min(
                        imgs, axis=(1, 2, 3), keepdims=True))
            assert np.all(np.min(imgs, axis=(1, 2, 3))
                          == [0.] * self.total_num)
            assert np.all(np.max(imgs, axis=(1, 2, 3))
                          == [1.] * self.total_num)

            np.random.seed(self.seed)
            ind = np.random.permutation(self.total_num)

            ind = ind[:self.train_num] if self.training else ind[self.train_num:]
            assert len(
                ind) == self.train_num if self.training else self.total_num - self.train_num
            logger.info('Shape of input {}'.format(imgs[ind, ...].shape))
            if self.dataset != 'stn_unlabeled':
                if self.training and cfg.TRAIN.CROP_RATIO < 1.:
                    imgs = utils.central_crop(imgs, cfg.TRAIN.CROP_RATIO)
                    counts = utils.central_crop(counts, cfg.TRAIN.CROP_RATIO)
                    self.image_shape = imgs.shape[1:]
                    logger.info('Input cropped. Shape {}'.format(
                        imgs[ind, ...].shape))
                self.data = tf.data.Dataset.from_generator(
                    lambda: zip(imgs[ind, ...], counts[ind, ...]),
                    (tf.float32, tf.float32),
                    (tf.TensorShape(self.image_shape), tf.TensorShape(self.image_shape)))
            else:
                self.data = tf.data.Dataset.from_generator(
                    lambda: zip(imgs[ind, ...], imgs[ind, ...]),
                    (tf.float32, tf.float32),
                    (tf.TensorShape(self.image_shape), tf.TensorShape(self.image_shape)))
        else:
            raise ValueError('Incorrect dataset')

    def preprocessing(self, augment, batch_size, num_epochs):
        def _parse(example_proto):
            # Create a dictionary describing the features.
            image_feature_description = {
                'image_raw': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.string),
                'depth': tf.io.FixedLenFeature([], tf.int64),
            }
            # Parse the input tf.Example proto using the dictionary above.
            parsed = tf.io.parse_single_example(
                example_proto, image_feature_description)
            image = tf.decode_raw(parsed['image_raw'], tf.uint8)
            image = tf.reshape(
                image, (self.image_shape[0], self.image_shape[1], parsed['depth']))
            image = tf.cast(image, tf.float32)
            image = tf.image.per_image_standardization(image)

            label = tf.decode_raw(parsed['label'], tf.float32)
            label = tf.reshape(
                label, (self.image_shape[0], self.image_shape[1], parsed['depth']))
            return image, label

        def _rand_flip(data, dim):
            assert data.get_shape().ndims == 4
            assert dim in [0, 1, 2]
            flip_flag = tf.random_uniform([]) > 0.5
            return tf.cond(flip_flag, lambda: tf.reverse(data, axis=[dim]), lambda: data)

        def _rand_rot(data):
            assert data.get_shape().ndims == 4
            data = tf.transpose(data, perm=[2, 0, 1, 3])
            data = tf.image.rot90(
                data, k=tf.random_uniform((), minval=0, maxval=2, dtype=tf.int32))
            return tf.transpose(data, perm=[1, 2, 0, 3])

        def _pad(image, label):
            data = tf.stack([image, label], axis=-1)
            assert data.get_shape().ndims == 4
            depth = tf.shape(data)[2]
            target_depth = cfg.TRAIN.PATCH_DEPTH * \
                tf.cast(tf.math.ceil(tf.math.truediv(
                    depth, cfg.TRAIN.PATCH_DEPTH)), tf.int32)
            depth_diff = tf.maximum(target_depth - depth, 0)
            paddings = tf.reshape(
                tf.stack([
                    0, 0, 0, 0, depth_diff // 2,
                    depth_diff - depth_diff // 2, 0, 0
                ]), [4, 2])
            data = tf.pad(data, paddings)
            data.set_shape(
                [512, 512, None, 2])
            return data

        # def _add_noise(image, label):
        #     assert image.get_shape().ndims == 4
        #     assert label.get_shape().ndims == 4
        #     val = tf.random_uniform([])
        #     noise_flag = val > 0.5
        #     img_shape = [cfg.TRAIN.PATCH_SIZE,
        #                  cfg.TRAIN.PATCH_SIZE, cfg.TRAIN.PATCH_DEPTH, 1]
        #     image = tf.cond(
        #         noise_flag,
        #         lambda: tf.random.normal(
        #             img_shape, mean=0.0, stddev=val) + image,
        #         lambda: image)
        #     return image, label
        def _add_noise_low(image, label):
            assert image.get_shape().ndims == 4
            assert label.get_shape().ndims == 4
            val = tf.random_uniform([]) * 0.1
            img_shape = [cfg.TRAIN.PATCH_SIZE,
                         cfg.TRAIN.PATCH_SIZE, cfg.TRAIN.PATCH_DEPTH, 1]
            image = tf.random.normal(img_shape, mean=0.0, stddev=val) + image
            return tf.clip_by_value(image, 0., 1.), label

        def _add_noise(image, label):
            assert image.get_shape().ndims == 4
            assert label.get_shape().ndims == 4
            val = tf.random_uniform([]) * 0.25  # 0.1
            img_shape = [cfg.TRAIN.PATCH_SIZE,
                         cfg.TRAIN.PATCH_SIZE, cfg.TRAIN.PATCH_DEPTH, 1]
            image = tf.random.normal(img_shape, mean=0.0, stddev=val) + image
            return tf.clip_by_value(image, 0., 1.), label

        def _add_noise_high(image, label):
            assert image.get_shape().ndims == 4
            assert label.get_shape().ndims == 4
            val = tf.random_uniform([]) * 0.4  # 0.1
            img_shape = [cfg.TRAIN.PATCH_SIZE,
                         cfg.TRAIN.PATCH_SIZE, cfg.TRAIN.PATCH_DEPTH, 1]
            image = tf.random.normal(img_shape, mean=0.0, stddev=val) + image
            return tf.clip_by_value(image, 0., 1.), label

        def _vary(image, label):
            assert image.get_shape().ndims == 4
            assert label.get_shape().ndims == 4

            def gaussian_prof(x, mu, sig):
                return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
            axis = np.random.randint(0, 3)
            dims = [cfg.TRAIN.PATCH_SIZE,
                    cfg.TRAIN.PATCH_SIZE, cfg.TRAIN.PATCH_DEPTH]
            intensity_factors = np.random.normal(1.0, 0.25, 2)
            center = np.random.randint(0, dims[axis] + 1)
            prof = gaussian_prof(np.linspace(
                1, dims[axis], dims[axis]), center, intensity_factors[0] * dims[axis])
            prof = prof * intensity_factors[1]
            if axis is 0:
                image = image * prof[:, np.newaxis, np.newaxis, np.newaxis]
            elif axis is 1:
                image = image * prof[np.newaxis, :, np.newaxis, np.newaxis]
            else:
                image = image * prof[np.newaxis, np.newaxis, :, np.newaxis]
            return tf.clip_by_value(image, 0., 1.), label

        def _mixup(image, label):
            # assert image.get_shape().ndims == 5
            # assert label.get_shape().ndims == 5
            # img_a, img_b = tf.unstack(image, num=2, axis=0)
            # label_a, label_b = tf.unstack(label, num=2, axis=0)
            # flag = tf.random_uniform([]) > 0.5
            # factor = tf.random_uniform([])
            # image = tf.cond(
            #     flag,
            #     lambda: factor * img_a + (1-factor) * img_b,
            #     lambda: img_a)
            # label = tf.cond(
            #     flag,
            #     lambda: factor * label_a + (1-factor) * label_b,
            #     lambda: label_a)
            # return tf.clip_by_value(image, 0., 1.), label
            return image, label

        def _random_brightness(image, label):
            assert image.get_shape().ndims == 4
            assert label.get_shape().ndims == 4
            image = tf.image.random_brightness(image, 0.5)
            return tf.clip_by_value(image, 0., 1.), label

        def _random_contrast(image, label):
            assert image.get_shape().ndims == 4
            assert label.get_shape().ndims == 4
            contrast_factor = tf.random.uniform([], 1.0, 4.0)
            image = (image - tf.reduce_mean(image)) * \
                contrast_factor + tf.reduce_mean(image)
            return tf.clip_by_value(image, 0., 1.), label

        def _random_sharpness(image, label):
            assert image.get_shape().ndims == 4
            assert label.get_shape().ndims == 4
            factor = tf.random.uniform([], 0.0, 4.0)
            image = tf.transpose(image, [2, 0, 1, 3])
            # SMOOTH PIL Kernel.
            kernel = (
                tf.constant(
                    [[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=tf.float32, shape=[3, 3, 1, 1]
                )
                / 13.0
            )
            strides = [1, 1, 1, 1]
            with tf.device('/cpu:0'):
                degenerate = tf.nn.depthwise_conv2d(
                    image, kernel, strides, padding="VALID", dilations=[1, 1])
            # For the borders of the resulting image, fill in the values of the
            # original image.
            degenerate = tf.clip_by_value(degenerate, 0., 1.)
            mask = tf.ones_like(degenerate)
            padded_mask = tf.pad(mask, [[0, 0], [1, 1], [1, 1], [0, 0]])
            padded_degenerate = tf.pad(
                degenerate, [[0, 0], [1, 1], [1, 1], [0, 0]])
            result = tf.where(tf.equal(padded_mask, 1),
                              padded_degenerate, image)
            # Blend
            image = result + (image - result) * factor
            image = tf.transpose(image, [1, 2, 0, 3])
            return tf.clip_by_value(image, 0., 1.), label

        def _cutout(image, label):
            assert image.get_shape().ndims == 4
            assert label.get_shape().ndims == 4
            replace = 0
            image_height = tf.shape(image)[0]
            image_width = tf.shape(image)[1]
            image_depth = tf.shape(image)[2]
            image = tf.image.per_image_standardization(tf.squeeze(image))
            image = tf.expand_dims(image, -1)
            label = tf.image.per_image_standardization(tf.squeeze(label))

            # Sample the center location in the image where the zero mask will be applied.
            cutout_center_height = tf.random.uniform(
                shape=[], minval=0, maxval=image_height,
                dtype=tf.int32)
            cutout_center_width = tf.random.uniform(
                shape=[], minval=0, maxval=image_width,
                dtype=tf.int32)
            cutout_center_depth = tf.random.uniform(
                shape=[], minval=0, maxval=image_depth,
                dtype=tf.int32)
            # The mask will be of size (2*pad_size x 2*pad_size x 2*pad_size_z).
            pad_size = tf.random.uniform(
                shape=[], minval=0, maxval=image_height//2, dtype=tf.int32)
            pad_size_z = tf.random.uniform(
                shape=[], minval=0, maxval=image_depth//2, dtype=tf.int32)

            lower_pad = tf.maximum(0, cutout_center_height - pad_size)
            upper_pad = tf.maximum(
                0, image_height - cutout_center_height - pad_size)
            left_pad = tf.maximum(0, cutout_center_width - pad_size)
            right_pad = tf.maximum(
                0, image_width - cutout_center_width - pad_size)
            front_pad = tf.maximum(0, cutout_center_depth - pad_size_z)
            rear_pad = tf.maximum(
                0, image_depth - cutout_center_depth - pad_size_z)

            cutout_shape = [image_height - (lower_pad + upper_pad),
                            image_width - (left_pad + right_pad),
                            image_depth - (front_pad + rear_pad)]
            padding_dims = [[lower_pad, upper_pad], [
                left_pad, right_pad], [front_pad, rear_pad]]
            mask = tf.pad(
                tf.zeros(cutout_shape, dtype=image.dtype),
                padding_dims, constant_values=1)
            mask = tf.expand_dims(mask, -1)
            image = tf.where(
                tf.equal(mask, 0),
                tf.ones_like(image, dtype=image.dtype) * replace,
                image)
            if self.dataset == 'stn_unlabeled':
                label = tf.stack([label, tf.squeeze(mask)], -1)
            else:
                label = tf.where(
                    tf.equal(mask, 0),
                    tf.ones_like(label, dtype=label.dtype) * replace,
                    label)
            return image, label

        def _augment(image, label):
            data = tf.stack([image, label], axis=-1)
            assert data.get_shape().ndims == 4
            data = tf.image.random_crop(
                data, [cfg.TRAIN.PATCH_SIZE, cfg.TRAIN.PATCH_SIZE, cfg.TRAIN.PATCH_DEPTH, 2])
            data.set_shape(
                [cfg.TRAIN.PATCH_SIZE, cfg.TRAIN.PATCH_SIZE, cfg.TRAIN.PATCH_DEPTH, 2])

            # flip xyz
            data = _rand_flip(data, 0)
            data = _rand_flip(data, 1)
            data = _rand_flip(data, 2)

            data = _rand_rot(data)
            image, label = tf.split(data, num_or_size_splits=2, num=2, axis=-1)
            return image, label

        def _tile(image, label):
            data = tf.stack([image, label], axis=-1)
            assert data.get_shape().ndims == 4

            if cfg.TRAIN.PATCH_SIZE != 200:
                tile_size = (cfg.TRAIN.PATCH_SIZE,
                             cfg.TRAIN.PATCH_SIZE, cfg.TRAIN.PATCH_DEPTH)
                data_shape = tf.shape(data)
                data = tf.transpose(
                    tf.reshape(data, [data_shape[0] // tile_size[0], tile_size[0],
                                      data_shape[1] // tile_size[1], tile_size[1],
                                      data_shape[2] // tile_size[2], tile_size[2], data_shape[3]]),
                    [0, 2, 4, 1, 3, 5, 6])
                data = tf.reshape(
                    data, [-1, tile_size[0], tile_size[1], tile_size[2], data_shape[3]])

            image, label = tf.split(data, num_or_size_splits=2, num=2, axis=-1)
            return image, label

        def _normalize(image, label):
            dim = 4 if cfg.TRAIN.PATCH_SIZE == 200 or self.training else 5
            assert image.get_shape().ndims == dim
            assert label.get_shape().ndims == dim
            image = tf.image.per_image_standardization(tf.squeeze(image))
            image = tf.expand_dims(image, -1)
            return image, label

        def _normalize_aug(image, label):
            assert image.get_shape().ndims == 4
            assert label.get_shape().ndims == 4
            image = tf.image.per_image_standardization(tf.squeeze(image))
            image = tf.expand_dims(image, -1)
            label = tf.image.per_image_standardization(tf.squeeze(label))
            label = tf.expand_dims(label, -1)
            return image, label

        if self.dataset == 'mbc':
            self.data = self.data.prefetch(
                buffer_size=tf.data.experimental.AUTOTUNE)
            self.data = self.data.map(
                _parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            self.data = self.data.map(
                _pad, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        self.data = self.data.cache()
        if augment:
            self.data = self.data.shuffle(buffer_size=self.train_num)

        self.data = self.data.repeat(num_epochs)
        if augment:
            self.data = self.data.map(
                _augment, num_parallel_calls=16)
            aug_dict = {'noise': _add_noise, 'noise_high': _add_noise_high,
                        'noise_low': _add_noise_low, 'brightness': _random_brightness,
                        'sharpness': _random_sharpness, 'contrast': _random_contrast,
                        'vary': _vary, 'mixup': _mixup, 'cutout': _cutout}

            if self.dataset in ['stn_75', 'stn_121', 'stn_eval']:
                for aug in cfg.TRAIN.SUP_AUG:
                    self.data = self.data.map(
                        aug_dict[aug], num_parallel_calls=16)

            elif self.dataset == 'stn_unlabeled':
                for aug in cfg.TRAIN.SEMI_AUG:
                    if aug != 'layer_noise' and aug != 'rot':
                        self.data = self.data.map(
                            aug_dict[aug], num_parallel_calls=16)
        else:
            self.data = self.data.map(
                _tile, num_parallel_calls=16)

        if self.dataset == 'stn_unlabeled':
            if 'cutout' not in cfg.TRAIN.SEMI_AUG:
                self.data = self.data.map(
                    _normalize_aug, num_parallel_calls=16)
        else:
            self.data = self.data.map(
                _normalize, num_parallel_calls=16)

        if self.training or cfg.TRAIN.PATCH_SIZE == 200:
            self.data = self.data.batch(batch_size)
        self.data = self.data.prefetch(buffer_size=10)
        dataset_iterator = self.data.make_one_shot_iterator()
        imgs, labels = dataset_iterator.get_next()
        return imgs, labels
