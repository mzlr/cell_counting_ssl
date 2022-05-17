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
    'stn_unlabeled': 'data/STN/unlabeled.hdf5',
    'vgg_unlabeled': 'data/VGG/VGG.hdf5', }

image_shape = {
    'vgg': (256, 256, 3),
    'dcc': (256, 256, 3),
    'mbm': (600, 600, 3),
    'adi': (152, 152, 3),
    'mbc': (512, 512),
    'stn_75': (224, 224, 64),
    'stn_121': (224, 224, 64),
    'stn_eval': (224, 224, 64),
    'stn_unlabeled': (256, 256, 64),
    'vgg_unlabeled': (256, 256, 3), }

train_num = {
    'vgg': 64,
    'dcc': 100,
    'mbm': 15,
    'adi': 50,
    'mbc': 58,
    'stn_75': 4,
    'stn_121': 4,
    'stn_eval': 0,
    'stn_unlabeled': 55,
    'vgg_unlabeled': 200, }

total_num = {
    'vgg': 200,
    'dcc': 176,
    'mbm': 44,
    'adi': 200,
    'mbc': 158,
    'stn_75': 16,
    'stn_121': 16,
    'stn_eval': 5,
    'stn_unlabeled': 55,
    'vgg_unlabeled': 200, }


class Dataset(object):
    def __init__(self, dataset, seed, training):
        if dataset not in ['vgg', 'dcc', 'mbm', 'adi', 'vgg_unlabeled']:
            raise ValueError('Wrong dataset name: {}'.format(dataset))
        self.dataset = dataset
        self.image_shape = image_shape[dataset]
        self.seed = seed
        self.training = training
        self.total_num = total_num[self.dataset]
        self.train_num = cfg.TRAIN.NUM

        self.data = {}
        with h5py.File(data_path[dataset], 'r') as hf:
            self.data['imgs'] = hf.get('imgs')[()]
            self.data['counts'] = hf.get('counts')[()]
        self.label_shape = self.image_shape[:2] + (1,)

        imgs = self.data['imgs'].astype(np.float32)
        counts = self.data['counts'].astype(np.float32)[..., np.newaxis]

        imgs = imgs / 255.
        assert np.max(imgs) <= 1
        assert np.min(imgs) >= 0

        assert imgs.shape == (self.total_num,) + self.image_shape
        assert counts.shape == (self.total_num,) + self.label_shape

        np.random.seed(self.seed)
        ind = np.random.permutation(self.total_num)

        split_num = 100
        mn = np.mean(imgs[ind[:split_num], ...], axis=(0, 1, 2))
        std = np.std(imgs[ind[:split_num], ...], axis=(0, 1, 2))
        if self.dataset == 'vgg_unlabeled':
            self.train = (imgs[ind[self.train_num:split_num], ...] - mn) / \
                std, (imgs[ind[self.train_num:split_num], ...] - mn) / std
        else:
            self.train = (imgs[ind[:self.train_num], ...] - mn) / \
                std, counts[ind[:self.train_num], ...]
        self.test = (imgs[ind[split_num:], ...] - mn) / \
            std, counts[ind[split_num:], ...]

    def preprocessing(self, augment, batch_size, num_epochs):
        def _augment(imgs, labels):
            inputs = tf.concat([imgs, labels], -1)
            assert inputs.get_shape().ndims == 3
            if self.image_shape == (600, 600, 3):
                inputs = tf.image.random_crop(inputs, [576, 576, 4])
            elif self.image_shape == (152, 152, 3):
                inputs = tf.image.random_crop(inputs, [144, 144, 4])
            elif self.image_shape == (256, 256, 3):
                if self.dataset == 'vgg_unlabeled':
                    inputs = tf.image.random_crop(inputs, [224, 224, 6])
                else:
                    inputs = tf.image.random_crop(inputs, [224, 224, 4])
            else:
                raise ValueError('Incorrect dataset')
            inputs = tf.image.random_flip_left_right(inputs)
            inputs = tf.image.random_flip_up_down(inputs)
            inputs = tf.image.rot90(
                inputs, k=tf.random_uniform((), minval=0, maxval=2, dtype=tf.int32))
            return inputs[:, :, :3], inputs[:, :, 3:]

        if self.training:
            dataset = tf.data.Dataset.from_tensor_slices(self.train)
        else:
            dataset = tf.data.Dataset.from_tensor_slices(self.test)

        if augment:
            dataset = dataset.map(
                _augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.shuffle(buffer_size=self.train_num)

        dataset = dataset.repeat(num_epochs).batch(batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        dataset_iterator = dataset.make_one_shot_iterator()
        imgs, labels = dataset_iterator.get_next()
        return imgs, labels
