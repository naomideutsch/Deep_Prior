import argparse
import os
from PIL import Image
import numpy as np
from preprocessing import blur_utils
import tensorflow as tf


def blur_preprocessing(args):
    kernel = blur_utils.get_kernel(args.kernel_size, args.sigma)
    images_paths = os.listdir(args.img_dir)
    for name in images_paths:
        image = Image.open(os.path.join(args.img_dir, name))
        tf_image = tf.convert_to_tensor(np.array(image).astype(np.float32))
        blr_image = blur_utils.apply_blur(tf_image, kernel)
        blurred = Image.fromarray(np.uint8(tf.Session().run(blr_image)))
        blurred.save(os.path.join(args.output_dir, name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--kernel-size', type=int, default=3)
    parser.add_argument('--sigma', type=int, default=2)
    args = parser.parse_args()
    if not os.path.exists(args.lr_imgs_dir):
        os.makedirs(args.lr_imgs_dir)
    blur_preprocessing(args)
