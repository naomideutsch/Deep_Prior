
import tensorflow as tf
import argparse
import numpy as np

def _gaussian_kernel(kernel_size, sigma, n_channels, dtype):
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
    g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
    g_norm2d = tf.pow(tf.reduce_sum(g), 2)
    g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
    g_kernel = tf.expand_dims(g_kernel, axis=-1)
    return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)

def _x_motion_kernel(kernel_size):
    np_result = np.zeros((kernel_size, kernel_size)).astype(np.float)
    np_result[:, kernel_size//2] = 1
    return tf.convert_to_tensor(np_result)

def _y_motion_kernel(kernel_size):
    np_result = np.zeros((kernel_size, kernel_size)).astype(np.float)
    np_result[kernel_size//2, :] = 1
    return tf.convert_to_tensor(np_result)




def get_kernel(kernel_size, sigma, type="gauss"):
    if type == "gauss":
        return lambda:  _gaussian_kernel(kernel_size, sigma, 3, tf.float32)
    if type == "x_motion":
        return lambda: _x_motion_kernel(kernel_size)
    if type == "y_motion":
        return lambda: _y_motion_kernel(kernel_size)


def apply_blur(img, kernel):
    blur = kernel()
    img = tf.nn.depthwise_conv2d(img, blur, [1,1,1,1], 'SAME')
    return img






