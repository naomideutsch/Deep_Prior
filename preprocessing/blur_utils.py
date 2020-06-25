
import tensorflow as tf
import argparse

def _gaussian_kernel(kernel_size, sigma, n_channels, dtype):
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
    g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
    g_norm2d = tf.pow(tf.reduce_sum(g), 2)
    g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
    g_kernel = tf.expand_dims(g_kernel, axis=-1)
    return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)


def get_kernel(kernel_size, sigma):
    return lambda:  _gaussian_kernel(kernel_size, sigma, 3, tf.float32)

def apply_blur(img, kernel):
    blur = kernel()
    print(blur)
    img = tf.nn.depthwise_conv2d(img[None], blur, [1,1,1,1], 'SAME')
    return img[0]



