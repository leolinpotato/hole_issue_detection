import json
import os
from natsort import natsorted
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--dir')
args = ap.parse_args()

predicted_dir = args.dir

sorted_dir = natsorted(os.listdir(predicted_dir))

total = len(sorted_dir)
hole_before = 0
hole_after = 0
total_r = 0
cnt = 0
with_hole_issue = 0

for file_name in sorted_dir:
	if file_name[0] == '.':
		continue
	predicted_file = open(f'{predicted_dir}/{file_name}', 'r')
	predicted_data = json.load(predicted_file)
	if predicted_data['hole_after_ASVS'] != 0:
		cnt += 1
		r = (predicted_data['hole_after_ASVS'] / predicted_data['product_after_ASVS']) / (predicted_data['hole_before_ASVS'] / predicted_data['product_before_ASVS'])
		hole_before += predicted_data['hole_before_ASVS']
		hole_after += predicted_data['hole_after_ASVS']
		total_r += r
	if r > 0.8:
		#print(file_name, r, predicted_data['hole_after_ASVS'], predicted_data['hole_before_ASVS'])
		with_hole_issue += 1
	predicted_file.close()


print("Result:")
print("Before:", hole_before / cnt)
print("After:", hole_after / cnt)
print("Ratio:", total_r / cnt)
print("Count:", cnt)
print("WithHoleIssue:", with_hole_issue)
