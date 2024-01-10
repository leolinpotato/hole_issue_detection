import cv2
import numpy as np
from numpy import linalg as LA
import ipdb
import argparse
import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchsummary import summary
import requests
from PIL import Image
from scipy.ndimage import zoom
from scipy.special import logsumexp
import  matplotlib.pyplot as plt
# import deepgaze_pytorch
import json
import os
from natsort import natsorted
from tqdm import tqdm

# command will be like: 
# python hole_detection.py --image "../../results_sd15/generated_image" --mask "../../results_sd15/mask" --output "../predicted_hole/VGG19"
ap = argparse.ArgumentParser()
ap.add_argument('--image', help='path of image')
ap.add_argument('--mask', help='path of mask')
ap.add_argument('--output', help='output dir name')
ap.add_argument('--debug', action='store_true', help='Enable debug mode')
args = ap.parse_args()

sz = 224

np.set_printoptions(threshold=np.inf)  # Display all array elements


### Below are different ways to get saliency map ###

def VGG19(image_path):
	image = cv2.imread(image_path)
	for param in model.parameters():
		param.requires_grad = False

	img = Image.open(image_path)


	# Preprocess the image
	def preprocess(image, size=224):
		transform = T.Compose([
			T.Resize((size, size)),
			T.ToTensor(),
			T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
			T.Lambda(lambda x: x[None]),
		])
		return transform(image)

	# preprocess the image
	X = preprocess(img)

	# we would run the model in evaluation mode
	model.eval()

	# we need to find the gradient with respect to the input image, so we need to call requires_grad_ on it
	X.requires_grad_()

	'''
    forward pass through the model to get the scores, note that VGG-19 model doesn't perform softmax at the end
    and we also don't need softmax, we need scores, so that's perfect for us.
    '''

	scores = model(X)

	# Get the index corresponding to the maximum score and the maximum score itself.
	score_max_index = scores.argmax()
	score_max = scores[0, score_max_index]

	'''
    backward function on score_max performs the backward pass in the computation graph and calculates the gradient of
    score_max with respect to nodes in the computation graph
    '''
	score_max.backward()

	'''
    Saliency would be the gradient with respect to the input image now. But note that the input image has 3 channels,
    R, G and B. To derive a single class saliency value for each pixel (i, j),  we take the maximum magnitude
    across all colour channels.
    '''
	saliency, _ = torch.max(X.grad.data.abs(), dim=1)

	# transform saliency tensor's shape and type
	resized_tensor = saliency[0].tolist()
	return np.array(resized_tensor)

'''
def load_deepgazeII():
	# you can use DeepGazeI or DeepGazeIIE
	model = deepgaze_pytorch.DeepGazeIIE(pretrained=True)

	# load precomputed centerbias log density (from MIT1003) over a 1024x1024 image
	# you can download the centerbias from https://github.com/matthias-k/DeepGaze/releases/download/v1.0.0/centerbias_mit1003.npy
	# alternatively, you can use a uniform centerbias via ``.
	centerbias_template = np.load('centerbias_mit1003.npy')
	return model, centerbias_template
def deepgazeII(image_path):
	image = cv2.resize(cv2.imread(image_path), (sz, sz))
	# rescale to match image size
	centerbias = zoom(centerbias_template, (image.shape[0]/centerbias_template.shape[0], image.shape[1]/centerbias_template.shape[1]), order=0, mode='nearest')
	# renormalize log density
	centerbias -= logsumexp(centerbias)

	image_tensor = torch.tensor([image.transpose(2, 0, 1)])
	centerbias_tensor = torch.tensor([centerbias])

	log_density_prediction = model(image_tensor, centerbias_tensor)

	image_np = log_density_prediction.squeeze().detach().numpy()
	saliency_map_normalized = cv2.normalize(image_np, None, 0, 1, cv2.NORM_MINMAX)
	return saliency_map_normalized
'''

def opencvSpectralResidual(image_path):
	image = cv2.resize(cv2.imread(image_path), (sz, sz))
	saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
	(success, saliencyMap) = saliency.computeSaliency(image)
	saliency_map_normalized = cv2.normalize(saliencyMap, None, 0, 1, cv2.NORM_MINMAX)
	return saliency_map_normalized

def opencvFineGrained(image_path):
	image = cv2.resize(cv2.imread(image_path), (sz, sz))
	saliency = cv2.saliency.StaticSaliencyFineGrained_create()
	(success, saliencyMap) = saliency.computeSaliency(image)
	saliency_map_normalized = cv2.normalize(saliencyMap, None, 0, 1, cv2.NORM_MINMAX)
	return saliency_map_normalized

### ###

def calculate_ASVS(saliencyMap, region):
	# remove unreasonable region
	for i in range(len(region)):
		for j in range(len(region[0])):
			region[i][j] = 1 if region[i][j] == 1 else 0
	matrix = saliencyMap * region / 255
	area = cv2.countNonZero(region)
	return pow(LA.norm(matrix), 2) / area

def get_mask(mask, DiEr=False):
	mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
	# revert color(0 to 1, 1 to 0)
	for i in range(len(mask)):
		for j in range(len(mask[0])):
			if mask[i][j] == 0:
				mask[i][j] = 1
			else:
				mask[i][j] = 0
	# try to capture hole which is not entirely closed
	mask_copy = mask.copy()
	if DiEr:
		kernel = np.ones((8, 8), np.uint8)
		mask_copy = cv2.dilate(mask_copy, kernel, iterations=1)
		mask_copy = cv2.erode(mask_copy, kernel, iterations=1)

	# find the border
	def compute_mask_border(mask):
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
	
	# border = compute_mask_border(mask_copy)
	
	# fill the pixel inside the border
	def fill_mask(mask):
	    # Create a copy of the mask to perform flood fill without modifying the original mask
	    mask_copy = mask.copy()

	    for i in range(len(mask_copy)):
	    	for j in range(len(mask_copy[0])):
	    		if i == 0 or i == len(mask_copy) - 1 or j == 0 or j == len(mask_copy[0]) - 1:
	    			mask_copy[i][j] = 0

	    # Perform flood fill starting from the outside point
	    cv2.floodFill(mask_copy, None, (0, 0), 2)

	    for i in range(len(mask_copy)):
	        for j in range(len(mask_copy[0])):
	            if mask_copy[i][j] == 2:
	                mask_copy[i][j] = 0
	            else:
	                mask_copy[i][j] = 1
	    return mask_copy
	
	filled_mask = fill_mask(mask_copy)

	# extend the mask
	def extend_mask_border(mask):
	    # Define a kernel for dilation
	    kernel = np.ones((int(sz / 10), int(sz / 10)), np.uint8)

	    # Perform dilation on the mask
	    extended_mask = cv2.dilate(mask, kernel, 1)

	    return extended_mask
	extended_mask = extend_mask_border(mask)

	return mask, filled_mask, extended_mask

def hole_issue_detection(image_path, mask_path):
	mask = cv2.resize(cv2.imread(mask_path), (sz, sz))
	omask, fmask, emask = get_mask(mask)
	bgmask = np.ones((sz, sz)) - omask
	# check whether contains hole
	hole = (fmask - omask).astype("uint8")
	for row in range(len(hole)):
		for i in range(len(hole[0])):
			if hole[row][i] != 1:
				hole[row][i] = 0
	area = np.sum(hole)
	print(area)
	if area < sz * sz / 300:  # without hole
		omask, fmask, emask = get_mask(mask, DiEr=True)
		bgmask = np.ones((sz, sz)) - omask
		# check whether contains hole
		hole = (fmask - omask).astype("uint8")
		for row in range(len(hole)):
			for i in range(len(hole[0])):
				if hole[row][i] != 1:
					hole[row][i] = 0
		area = np.sum(hole)
		print('enclose area: ', area)
		if area < sz * sz / 100:
			return 0

	# have hole, need to compute ASVS to check have issue or not
	kernel = np.ones((8, 8), np.uint8)
	hole_erode = cv2.erode(hole, kernel, iterations=1)
	hole_edge = hole - hole_erode
	# cv2.imshow('hole', hole * 255)
	# cv2.imshow('hole_edge', hole_edge * 255)
	# cv2.waitKey(0)

	# saliencyMap = function(), can pass in different function, and the return value is normalized between 0 and 1
	saliencyMap = opencvFineGrained(image_path)
	saliencyMap = (saliencyMap * 255).astype('uint8')
	border = (emask - fmask).astype("uint8")

	for row in range(len(border)):
		for i in range(len(border[0])):
			if border[row][i] != 1:
				border[row][i] = 0

	hist_hole = cv2.calcHist([saliencyMap], [0], hole, [256], [0, 256])
	hist_border = cv2.calcHist([saliencyMap], [0], border, [256], [0, 256])
	# X_axis = np.arange(256)
	# plt.bar(X_axis - 0.2, hist_hole.reshape(-1), color='b')
	# plt.bar(X_axis + 0.2, hist_border.reshape(-1), color='g')
	# plt.show()

	similarity = cv2.compareHist(hist_hole, hist_border, cv2.HISTCMP_CORREL)

	hole_ASVS = calculate_ASVS(saliencyMap, hole)
	border_ASVS = calculate_ASVS(saliencyMap, border)
	r = 1.3
	t = 0.03
	return 1 if hole_ASVS / border_ASVS > r or hole_ASVS - border_ASVS > t else 0  # 1 means with hole_issue, 0 means without hole_issue
	# return similarity, hole_ASVS, border_ASVS

# only need to change function used in "hole_issue_detection" and the name of the result file
def main():
	image = args.image  # the directory of generated_images
	mask = args.mask
	sorted_image = natsorted(os.listdir(image))
	sorted_mask = natsorted(os.listdir(mask))
	cnt = -1
	for file_name in tqdm(sorted_image, desc="Processing files", total=len(sorted_image)):
		if f'{file_name.split("_")[0]}.png' not in sorted_mask or file_name[0] == '.':
			continue
		cnt += 1
		# obj_name = file_name.split('_')[0]  # get the object name, "bag_1.png" we get "bag"
		if args.debug and cnt % 10:  # if debugging, only choose 1/40 of images
			continue
		with open(f'{args.output}/{file_name}.json', 'w') as f:
			print(f'{image}/{file_name}', f'{mask}/{file_name}')
			value = hole_issue_detection(f'{image}/{file_name}', f'{mask}/{file_name.split("_")[0]}.png')
			# similarity, hole_ASVS, border_ASVS = hole_issue_detection(f'{image}/{file_name}', f'{mask}/{file_name.split("_")[0]}.png')
			# if similarity > 0.5:  # 0.5 is a threshold, which can be modified
			# 	hole_issue = False
			# else:
			# 	hole_issue = True
			# data = {
			# 	"product_hole_issue": hole_issue,
			# 	"similarity": similarity
			# }
			# data["product_hole_issue"] = bool(data["product_hole_issue"])
			# data = {
			# 	"similarity": similarity,
			# 	"hole_ASVS": hole_ASVS,
			# 	"border_ASVS": border_ASVS,
			# }
			data = {
				"hole_issue": value,
			}
			json.dump(data, f, ensure_ascii=False, indent=2)
		if args.debug:
			break

# model, centerbias_template = load_deepgazeII()	
# model = torchvision.models.vgg19(pretrained=True)
main()