import json
import os
from natsort import natsorted
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--dir')
args = ap.parse_args()

labeled_dir = args.dir
arr = np.zeros(2000)


for directory in os.listdir(labeled_dir):
	if directory[0] == '.':
		continue
	sorted_dir = natsorted(os.listdir(f'{labeled_dir}/{directory}'))

	for file_name in sorted_dir:
		labeled_idx = file_name.split('.')[0]
		labeled_file = open(f'{labeled_dir}/{directory}/{file_name}', 'r')
		labeled_data = json.load(labeled_file)
		if labeled_data['product_hole_issue']:
			arr[int(labeled_idx)] += 1
		labeled_file.close()


with open(f'../result/hole_issue.json', 'w') as j:
	data = []
	for i in range(2000):
		if arr[i] > 1:
			data.append(i)
	json.dump(data, j, ensure_ascii=False, indent=2)

