import tensorflow as tf
import numpy as np
import PIL.Image
from matplotlib import pyplot as P
import cv2
from numpy import linalg as LA
import argparse
import json
import os
from natsort import natsorted
from tqdm import tqdm
from scipy.ndimage import zoom
from scipy.special import logsumexp
import torch
from pysaliency.plotting import visualize_distribution
import deepgaze_pytorch

# command will be like: 
# python model.py --image "../../results_sd15/generated_image" --mask "../../results_sd15/mask" --output "../predicted_hole/VGG19"
ap = argparse.ArgumentParser()
ap.add_argument('--image', help='path of image')
ap.add_argument('--mask', help='path of mask')
ap.add_argument('--output', help='output dir name')
ap.add_argument('--debug', action='store_true', help='Enable debug mode')
args = ap.parse_args()

w = 512
h = 512

def ShowImage(im, title='', ax=None):
  if ax is None:
    P.figure()
  P.axis('off')
  P.imshow(im)
  P.title(title)

def ShowGrayscaleImage(im, title='', ax=None):
  if ax is None:
    P.figure()
  P.axis('off')

  P.imshow(im, cmap=P.cm.gray, vmin=0, vmax=1)
  P.title(title)

def ShowHeatMap(im, title, ax=None):
  if ax is None:
    P.figure()
  P.axis('off')
  P.imshow(im, cmap='inferno')
  P.title(title)

def LoadImage(file_path):
  im = PIL.Image.open(file_path)
  im = im.resize((w,h))
  im = np.asarray(im)
  if im.dtype != np.uint8:
  	im = im.astype(np.uint8)
  return im

def PreprocessImage(im):
  im = tf.keras.applications.vgg16.preprocess_input(im)
  return im

def normalize(image):
	return (image - image.min()) / (image.max() - image.min())

def CalASVS(saliencyMap, region):
	# remove unreasonable region
	for i in range(len(region)):
		for j in range(len(region[0])):
			region[i][j] = 1 if region[i][j] == 1 else 0
	matrix = saliencyMap * region
	area = cv2.countNonZero(region)
	return pow(LA.norm(matrix), 2) / area


def GetMask(mask, DiEr=False):
	if len(mask.shape) == 3 and mask.shape[2] != 1:
		mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)	# revert color(0 to 1, 1 to 0)
	mask = np.where(mask == 0, 1, 0)
	mask = mask.astype(np.uint8)

	# try to capture hole which is not entirely closed
	mask_copy = mask.copy()
	if DiEr:
		kernel = np.ones((8, 8), np.uint8)
		mask_copy = cv2.dilate(mask_copy, kernel, iterations=1)
		mask_copy = cv2.erode(mask_copy, kernel, iterations=1)

	# find the border
	def ComputeMaskBorder(mask):
		# Convert the mask to an 8-bit single-channel image (0 and 255 values)
		mask = (mask * 255).astype(np.uint8)

		# Find contours in the mask
		contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		# Draw the contours on a black background
		border = np.zeros_like(mask)

		# Draw all the found contours
		cv2.drawContours(border, contours, -1, (255), thickness=1)

		# Convert the border to a binary image (0s and 1s)
		border = (border > 0).astype(np.uint8)

		return border
	
	# border = ComputeMaskBorder(mask_copy)
	
	# fill the pixel inside the border
	def FillMask(mask):
	    # Create a copy of the mask to perform flood fill without modifying the original mask
	    mask_copy = mask.copy()

	    mask_copy[0, :] = 0  # Set the first row to 0
	    mask_copy[-1, :] = 0  # Set the last row to 0
	    mask_copy[:, 0] = 0  # Set the first column to 0
	    mask_copy[:, -1] = 0  # Set the last column to 0
	    
	    # Perform flood fill starting from the outside point
	    cv2.floodFill(mask_copy, None, (0, 0), 2)

	    mask_copy = np.where(mask_copy == 2, 0, 1)

	    return mask_copy
	
	filled_mask = FillMask(mask_copy)

	# extend the mask
	def ExtendMaskBorder(mask):
	    # Define a kernel for dilation
	    kernel = np.ones((int(w / 10), int(h / 10)), np.uint8)

	    # Perform dilation on the mask
	    extended_mask = cv2.dilate(mask, kernel, 1)

	    return extended_mask
	extended_mask = ExtendMaskBorder(mask)

	return mask, filled_mask, extended_mask

def LoadDeepgazeIIE():
	# you can use DeepGazeI or DeepGazeIIE
	model = deepgaze_pytorch.DeepGazeIIE(pretrained=True)

	# load precomputed centerbias log density (from MIT1003) over a 1024x1024 image
	# you can download the centerbias from https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy
	# alternatively, you can use a uniform centerbias via ``.
	centerbias_template = np.load('centerbias_mit1003.npy')
	return model, centerbias_template
def DeepGazeIIE(image_path):
	im_orig = LoadImage(image_path)
	# rescale to match image size
	centerbias = zoom(centerbias_template, (im_orig.shape[0]/centerbias_template.shape[0], im_orig.shape[1]/centerbias_template.shape[1]), order=0, mode='nearest')
	# renormalize log density
	centerbias -= logsumexp(centerbias)

	image_tensor = torch.tensor(im_orig.transpose(2, 0, 1)[np.newaxis, ...])
	centerbias_tensor = torch.tensor(centerbias[np.newaxis, ...])

	log_density_prediction = model(image_tensor, centerbias_tensor)

	image_np = log_density_prediction.detach().cpu().numpy()[0, 0]
	saliency_map_normalized = cv2.normalize(image_np, None, 0, 1, cv2.NORM_MINMAX)

	# Set up matplot lib figures.
	# ROWS = 1
	# COLS = 2
	# UPSCALE_FACTOR = 10
	# P.figure(figsize=(ROWS * UPSCALE_FACTOR, COLS * UPSCALE_FACTOR))

	# Show the image
	# ShowImage(im_orig, ax=P.subplot(ROWS, COLS, 1))
	# ShowImage(image_np, ax=P.subplot(ROWS, COLS, 2))

	# f, axs = P.subplots(nrows=1, ncols=3, figsize=(12, 3))
	# axs[0].imshow(im_orig)
	# axs[0].set_axis_off()
	# axs[1].matshow(log_density_prediction.detach().cpu().numpy()[0, 0])  # first image in batch, first (and only) channel
	# axs[1].set_axis_off()
	# visualize_distribution(log_density_prediction.detach().cpu().numpy()[0, 0], ax=axs[2])
	# axs[2].set_axis_off()

	# Render the saliency masks.
	#ShowGrayscaleImage(saliency_map_normalized, ax=P.subplot(ROWS, COLS, 2))

	return saliency_map_normalized

def HoleIssueDetection(image_path, mask_path):
	mask = LoadImage(mask_path)
	omask, fmask, emask = GetMask(mask)
	bgmask = np.ones((w, h), np.uint8) - omask
	# check whether contains hole
	hole = (fmask - omask).astype("uint8")
	hole = np.where(hole != 1, 0, hole)
	area = np.sum(hole)
	if area < w * h / 300:  # without hole
		omask, fmask, emask = GetMask(mask, DiEr=True)
		bgmask = np.ones((w, h)) - omask
		# check whether contains hole
		hole = (fmask - omask).astype("uint8")
		hole = np.where(hole != 1, 0, hole)
		area = np.sum(hole)
		print('enclose area: ', area)
		if area < w * h / 100:
			return 0, 0

	saliencyMap = DeepGazeIIE(image_path)

	border = (emask - fmask).astype("uint8")
	border = np.where(border != 1, 0, border)

	hole_ASVS = CalASVS(saliencyMap, hole)
	border_ASVS = CalASVS(saliencyMap, border)

	return hole_ASVS, border_ASVS

def main():
	# the directory of generated_images/mask
	image = args.image  
	mask = args.mask
	sorted_image = natsorted(os.listdir(image))
	sorted_mask = natsorted(os.listdir(mask))

	for file_name in tqdm(sorted_image, desc="Processing files", total=len(sorted_image)):
		if f'{file_name.split("_")[0]}.png' not in sorted_mask or file_name[0] == '.':
			continue
		if args.debug:  # only do once
			hole_ASVS, border_ASVS = HoleIssueDetection(f'{image}/{file_name}', f'{mask}/{file_name.split("_")[0]}.png')
			print(hole_ASVS, border_ASVS)
			break
		with open(f'{args.output}/{file_name}.json', 'w') as f:
			print(f'{image}/{file_name}', f'{mask}/{file_name}')
			hole_ASVS, border_ASVS = HoleIssueDetection(f'{image}/{file_name}', f'{mask}/{file_name.split("_")[0]}.png')
			data = {
				"hole_ASVS": hole_ASVS,
				"border_ASVS": border_ASVS
			}
			json.dump(data, f, ensure_ascii=False, indent=2)
		
	#P.show()

model, centerbias_template = LoadDeepgazeIIE()
main()