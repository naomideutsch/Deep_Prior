



import argparse
from PIL import Image





# def blur(image_path, output_path, blur_function):
# 	image =











if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--img-dir', type=str, required=True)
	parser.add_argument('--lr-imgs-dir', type=str, required=True)
	parser.add_argument('--blur-type', type=str, required=True)

	args = parser.parse_args()