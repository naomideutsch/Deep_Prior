
import tensorflow as tf
import argparse
import numpy as np
from skimage.draw import circle
import cv2
import pickle
import os



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
        return lambda image: cv2.bilateralFilter(cv2.UMat(image),kernel_size,sigma,sigma) # sigma = 75
    if type == "psf":
        return lambda image: PsfBlur_random(image) # sigma = 75


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





def PsfBlur(image, psfid):
    kernel = psfDictionary[psfid]
    convolved = apply_blur(image, kernel)
    # convolved = convolve2d(imgarray, kernel, mode='same', fillvalue=255.0).astype("uint8")
    img = Image.fromarray(convolved)
    return img


def PsfBlur_random(img):
    pickledPsfFilename = os.path.join(os.path.dirname(__file__), "psf.pkl")

    with open(pickledPsfFilename, 'rb') as pklfile:
        psfDictionary = pickle.load(pklfile)
    psfid = np.random.randint(0, len(psfDictionary))
    kernel = psfDictionary[psfid]
    convolved = apply_blur(img, kernel)


    return convolved









