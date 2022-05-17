import logging
import os
from collections import OrderedDict
import tensorflow as tf
import h5py
import numpy as np


class AttrDict(dict):
    """ subclass dict and define getter-setter. This behaves as both dict and obj"""

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value


def setup_custom_logger(name):
    formatter = logging.Formatter(
        '[%(levelname)s: %(filename)s:%(lineno)d]: %(message)s')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    logger.addHandler(handler)
    logger.propagate = False
    return logger


def remove_first_scope(name):
    return '/'.join(name.split('/')[1:])


def collect_vars(scope, start=None, end=None, prepend_scope=None):
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
    var_dict = OrderedDict()
    if isinstance(start, str):
        for i, var in enumerate(vars):
            var_name = remove_first_scope(var.op.name)
            if var_name.startswith(start):
                start = i
                break
    if isinstance(end, str):
        for i, var in enumerate(vars):
            var_name = remove_first_scope(var.op.name)
            if var_name.startswith(end):
                end = i
                break
    # Using None for one of the slice parameters is the same as omitting it.
    for var in vars[start:end]:
        var_name = remove_first_scope(var.op.name)
        if prepend_scope is not None:
            var_name = os.path.join(prepend_scope, var_name)
        var_dict[var_name] = var
    return var_dict


def save_results(file_path, file_dict):
    with h5py.File(file_path, 'w') as hf:
        for k in file_dict:
            g = hf.create_group(k)
            for i, item in enumerate(file_dict[k]):
                g.create_dataset(str(i), data=item)


def tversky_loss(y_true, y_pred, weights):
    y_true = tf.layers.flatten(y_true)
    y_pred = tf.layers.flatten(y_pred)
    weights = tf.layers.flatten(weights) if weights != 1. else weights

    alpha = 0.7
    smooth = 1.0
    true_pos = tf.reduce_sum(weights * y_true * y_pred, axis=-1)
    false_neg = tf.reduce_sum(weights * y_true * (1-y_pred), axis=-1)
    false_pos = tf.reduce_sum(weights * (1-y_true) * y_pred, axis=-1)
    tversky = (true_pos + smooth) /\
        (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)
    return tf.reduce_mean(1-tversky)


def kl_divergence(p, q, weights):
    p = tf.layers.flatten(p)
    q = tf.layers.flatten(q)
    weights = tf.layers.flatten(weights) if weights != 1. else weights
    epsilon = 1e-7
    p = tf.clip_by_value(p * weights, epsilon, 1.)
    q = tf.clip_by_value(q * weights, epsilon, 1.)
    return tf.reduce_mean(p * tf.math.log(p / q))


def anneal_sup_loss(sup_logits, sup_labels, global_step):
    global_step = tf.cast(global_step, tf.float32)
    ratio = tf.clip_by_value(global_step / 1550, 0., 1.)
    threshold = 1 - tf.exp(5.0 * (ratio-1)) * (1 - 0.05)
    tf.summary.scalar('sup_threshold', threshold)
    distance = tf.abs(sup_logits - sup_labels) / tf.maximum(sup_labels, 1e-7)
    more_than_threshold = tf.greater(distance, threshold)
    weight = tf.stop_gradient(tf.cast(more_than_threshold, tf.float32))
    tf.summary.scalar('sup_ratio', tf.reduce_mean(weight))
    return weight


def central_crop(image, central_fraction):
    assert len(image.shape) == 4

    def get_start_size(x, f=np.cbrt(central_fraction)):
        s = int((x - x * f) / 2.)
        return s, x - s
    h, w, d = map(get_start_size, image.shape[1:])
    return image[:, h[0]:h[1], w[0]:w[1], d[0]:d[1]]


def gauss3D(shape=(3, 3, 3), sigma=0.5):
    """
    3D gaussian mask - should give the same result as MATLAB's
    fspecial3('gaussian',[shape],[sigma])
    """
    assert len(shape) == 3
    m, n, o = [(ss-1.)/2. for ss in shape]
    y, x, z = np.ogrid[-m:m+1, -n:n+1, -o:o+1]
    h = np.exp(-(x*x + y*y + z*z) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def gauss2D(shape=(3, 3), sigma=0.5):
    """
    3D gaussian mask - should give the same result as MATLAB's
    fspecial3('gaussian',[shape],[sigma])
    """
    assert len(shape) == 2
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def _ssim_helper(x, y, reducer, max_val, compensation=1.0, k1=0.01, k2=0.03):
    """Helper function for computing SSIM.
    SSIM estimates covariances with weighted sums.  The default parameters
    use a biased estimate of the covariance:
    Suppose `reducer` is a weighted sum, then the mean estimators are
        \mu_x = \sum_i w_i x_i,
        \mu_y = \sum_i w_i y_i,
    where w_i's are the weighted-sum weights, and covariance estimator is
    cov_{xy} = \sum_i w_i (x_i - \mu_x) (y_i - \mu_y)
    with assumption \sum_i w_i = 1. This covariance estimator is biased, since
    E[cov_{xy}] = (1 - \sum_i w_i ^ 2) Cov(X, Y).
    For SSIM measure with unbiased covariance estimators, pass as `compensation`
    argument (1 - \sum_i w_i ^ 2).
    Arguments:
        x: First set of images.
        y: Second set of images.
        reducer: Function that computes 'local' averages from the set of images. For
          non-convolutional version, this is usually tf.reduce_mean(x, [1, 2]), and
          for convolutional version, this is usually tf.nn.avg_pool2d or
          tf.nn.conv2d with weighted-sum kernel.
        max_val: The dynamic range (i.e., the difference between the maximum
          possible allowed value and the minimum allowed value).
        compensation: Compensation factor. See above.
        k1: Default value 0.01
        k2: Default value 0.03 (SSIM is less sensitivity to K2 for lower values, so
          it would be better if we took the values in the range of 0 < K2 < 0.4).
    Returns:
        A pair containing the luminance measure, and the contrast-structure measure.
    """

    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2

    # SSIM luminance measure is
    # (2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1).
    mean0 = reducer(x)
    mean1 = reducer(y)
    num0 = mean0 * mean1 * 2.0
    den0 = tf.square(mean0) + tf.square(mean1)
    luminance = (num0 + c1) / (den0 + c1)

    # SSIM contrast-structure measure is
    #   (2 * cov_{xy} + c2) / (cov_{xx} + cov_{yy} + c2).
    # Note that `reducer` is a weighted sum with weight w_k, \sum_i w_i = 1, then
    #   cov_{xy} = \sum_i w_i (x_i - \mu_x) (y_i - \mu_y)
    #          = \sum_i w_i x_i y_i - (\sum_i w_i x_i) (\sum_j w_j y_j).
    num1 = reducer(x * y) * 2.0
    den1 = reducer(tf.square(x) + tf.square(y))
    c2 *= compensation
    cs = (num1 - num0 + c2) / (den1 - den0 + c2)

    # SSIM score is the product of the luminance and contrast-structure measures.
    return luminance, cs


def ssim(img1, img2,
         max_val=1.0,
         filter_size=11,
         filter_sigma=1.5,
         k1=0.01, k2=0.03, weights=1.):
    """Computes SSIM index between img1 and img2 per color channel.
    This function matches the standard SSIM implementation from:
    Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image
    quality assessment: from error visibility to structural similarity. IEEE
    transactions on image processing.
    Details:
    - 11x11 Gaussian filter of width 1.5 is used.
    - k1 = 0.01, k2 = 0.03 as in the original paper.
    Args:
        img1: First image batch.
        img2: Second image batch.
        max_val: The dynamic range of the images (i.e., the difference between the
          maximum the and minimum allowed values).
        filter_size: Default value 11 (size of gaussian filter).
        filter_sigma: Default value 1.5 (width of gaussian filter).
        k1: Default value 0.01
        k2: Default value 0.03 (SSIM is less sensitivity to K2 for lower values, so
          it would be better if we took the values in the range of 0 < K2 < 0.4).
        weights: weights for ssim
    Returns:
        A pair of tensors containing and channel-wise SSIM and contrast-structure
        values. The shape is [..., channels].
    """
    assert img1.get_shape().ndims == 5
    assert img2.get_shape().ndims == 5
    kernel = gauss3D((filter_size, filter_size, filter_size),
                     filter_sigma)[..., np.newaxis, np.newaxis]
    assert kernel.shape == (filter_size, filter_size, filter_size, 1, 1)
    kernel = tf.constant(kernel, tf.float32)

    # The correct compensation factor is `1.0 - tf.reduce_sum(tf.square(kernel))`,
    # but to match MATLAB implementation of MS-SSIM, we use 1.0 instead.
    compensation = 1.0

    def reducer(x):
        return tf.nn.conv3d(x, kernel, strides=[1, 1, 1, 1, 1], padding='SAME')

    luminance, cs = _ssim_helper(
        img1, img2, reducer, max_val, compensation, k1, k2)
    # loss is 1 - ssim \in [0, 1]
    ssim_loss = 1 - luminance * cs

    # Average over [batch, height, width, depth, channel]
    ssim_val = tf.losses.compute_weighted_loss(ssim_loss, weights=weights)
    return ssim_val


def set_flags():
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_CACHE_DISABLE'] = '1'
    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
    os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
    os.environ['TF_ADJUST_HUE_FUSED'] = '1'
    os.environ['TF_ADJUST_SATURATION_FUSED'] = '1'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
    os.environ['TF_SYNC_ON_FINISH'] = '0'
    os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'
    os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'
    os.environ['TF_ENABLE_NHWC'] = '1'
    # NVIDIA Tools Extension, only for debugging and profiling
    os.environ['TF_DISABLE_NVTX_RANGES'] = '1'
    # os.environ['TF_GPU_HOST_MEM_LIMIT_IN_MB'] = '200000'
    # os.environ['TF_XLA_FLAGS'] = '--tf_xla_always_defer_compilation=true'
