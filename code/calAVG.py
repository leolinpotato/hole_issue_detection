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

def main():
	mask = '../predicted_hole/hole_issue/1'
	total = 0
	avg_ratio = 0
	avg_similarity = 0
	avg_diff = 0
	have_issue = 0
	sorted_mask = natsorted(os.listdir(mask))

	for file_name in tqdm(sorted_mask, desc="Processing files", total=len(sorted_mask)):
		if file_name[0] == '.':
			continue
		file = open(f'{mask}/{file_name}', 'r')
		data = json.load(file)
		r = 1.3  # ratio
		t = 0.03 # threshold
		s = 0.3  # similarity
		# if (float(data['similarity']) < s) or (abs(float(data['border_ASVS'] - float(data['hole_ASVS']))) > t):
		# 	have_issue += 1
		B = float(data['border_ASVS'])
		H = float(data['hole_ASVS'])
		S = float(data['similarity'])

		avg_ratio += H / B if H > B else B / H
		avg_similarity += S
		avg_diff += abs(H - B)

		cr = 1
		cs = -5
		cd = 20

		score = (H / B if H > B else B / H) * cr + S * cs + abs(H - B) * cd

		if H / B > r or H - B > t:
			have_issue += 1

		total += 1

	print(have_issue / total)
	print('avg_similarity: ', avg_similarity / total)
	print('avg_diff: ', avg_diff / total)
	print('avg_ratio: ', avg_ratio / total)

main()