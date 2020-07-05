
import tensorflow as tf
import argparse
import numpy as np
from skimage.draw import circle
import cv2
import pickle
import os
import math



def _gaussian_kernel(kernel_size, sigma, n_channels, dtype):
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
    g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
    g_norm2d = tf.pow(tf.reduce_sum(g), 2)
    g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
    g_kernel = tf.expand_dims(g_kernel, axis=-1)
    return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)

def _x_motion_kernel(kernel_size):
    np_result = np.zeros((kernel_size, kernel_size,3,1)).astype(np.float32)
    np_result[:, kernel_size//2, :, :] = 1
    return tf.convert_to_tensor(np_result * (1/kernel_size))

def _y_motion_kernel(kernel_size):
    np_result = np.zeros((kernel_size, kernel_size,3,1)).astype(np.float32)
    np_result[kernel_size//2, :, :, :] = 1
    return tf.convert_to_tensor(np_result* (1/kernel_size))



def _disk_kernel(kernel_size):

    kernelwidth = kernel_size
    kernel = np.zeros((kernelwidth, kernelwidth, 3 ,1), dtype=np.float32)
    circleCenterCoord = kernel_size // 2
    circleRadius = circleCenterCoord + 1

    rr, cc = circle(circleCenterCoord, circleCenterCoord, circleRadius)
    kernel[rr, cc, :, :] = 1

    if (kernel_size == 3 or kernel_size == 5):
        kernel = Adjust(kernel, kernel_size)

    normalizationFactor = np.count_nonzero(kernel[:,:,0,0])
    kernel = kernel / normalizationFactor
    return kernel


def Adjust(kernel, kernelwidth):
    kernel[0, 0] = 0
    kernel[0, kernelwidth - 1] = 0
    kernel[kernelwidth - 1, 0] = 0
    kernel[kernelwidth - 1, kernelwidth - 1] = 0
    return kernel




def get_blur(kernel_size, sigma, type="gauss"):
    if type == "gauss":
        kernel = _gaussian_kernel(kernel_size, sigma, 3, tf.float32) #sigma = 2
        return lambda image:  apply_blur(image, kernel)
    if type == "x_motion":
        kernel = _x_motion_kernel(kernel_size)
        return lambda image: apply_blur(image, kernel)
    if type == "y_motion":
        kernel = _y_motion_kernel(kernel_size)
        return lambda image: apply_blur(image, kernel)

    if type == "disk":
        kernel = _disk_kernel(kernel_size)
        return lambda image: apply_blur(image, kernel)
    if type == "median":
        return lambda image: cv2.medianBlur(image, kernel_size)

    if type == "bi":
        return lambda image: bilateral_filter(image,kernel_size,sigma,sigma) # sigma = 75



def apply_blur(img, kernel):
    pointwise_filter = tf.eye(3, batch_shape=[1, 1])
    img = tf.nn.separable_conv2d(img, kernel, pointwise_filter, [1,1,1,1], 'SAME')
    return img


def convert_to_gray(image):

    gray = 0.2126 * image[:, :, :, 0] + 0.7152 * image[:, :, :, 1] + 0.0722 * image[:, :, :, 2]
    gray = gray[:,:,:, None]
    result = tf.concat([gray, gray], -1)
    result = tf.concat([result, gray], -1)
    final = result.eval(session=tf.compat.v1.Session())


    return final


def distance(x, y, i, j):
    return np.sqrt((x-i)**2 + (y-j)**2)


def gaussian(x, sigma):
    return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x ** 2) / (2 * sigma ** 2))


def apply_bilateral_filter(source, filtered_image, x, y, diameter, sigma_i, sigma_s):
    hl = diameter/2
    i_filtered = 0
    Wp = 0
    i = 0
    while i < diameter:
        j = 0
        while j < diameter:
            neighbour_x = x - (hl - i)
            neighbour_y = y - (hl - j)
            if neighbour_x >= len(source):
                neighbour_x -= len(source)
            if neighbour_y >= len(source[0]):
                neighbour_y -= len(source[0])
            gi = gaussian(source[neighbour_x][neighbour_y] - source[x][y], sigma_i)
            gs = gaussian(distance(neighbour_x, neighbour_y, x, y), sigma_s)
            w = gi * gs
            i_filtered += source[neighbour_x][neighbour_y] * w
            Wp += w
            j += 1
        i += 1
    i_filtered = i_filtered / Wp
    filtered_image[x][y] = int(round(i_filtered))


def bilateral_filter(source, filter_diameter, sigma_i, sigma_s):
    filtered_image = np.zeros(source.shape)

    i = 0
    while i < len(source):
        j = 0
        while j < len(source[0]):
            apply_bilateral_filter(source, filtered_image, i, j, filter_diameter, sigma_i, sigma_s)
            j += 1
        i += 1
    return filtered_image








