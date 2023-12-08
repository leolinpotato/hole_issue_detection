import json
import os
from natsort import natsorted
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--dir')
ap.add_argument('--r', type=float, help='ratio')
args = ap.parse_args()

predicted_dir = args.dir

sorted_dir = natsorted(os.listdir(predicted_dir))
hole_issue_file = open('../result/hole_issue.json', 'r')
hole_issue_data = json.load(hole_issue_file)

total = len(sorted_dir)
true_to_true = 0
false_to_true = 0

for file_name in sorted_dir:
	predicted_file = open(f'{predicted_dir}/{file_name}', 'r')
	predicted_data = json.load(predicted_file)
	if predicted_data['product_hole_issue']:
		if int(file_name.split('.')[0]) in hole_issue_data:
			true_to_true += 1
		else:
			false_to_true += 1
	predicted_file.close()

hole_issue_file.close()

print(true_to_true)
print(false_to_true)

# with open(f'../result/{args.dir}_r={args.r}.json', 'w') as j:
# 	data = {
# 		'Total_num': total,
# 		'Without hole issue': total_false,
# 		'With hole issue': total_true,
# 		'error rate': (true_to_false + false_to_true) / total,
# 		'Without to With': false_to_true,
# 		'With to Without': true_to_false,
# 		'Without to With rt': false_to_true / total_false,
# 		'With to Without rt': true_to_false / total_true,
# 		'Without to With file name': false_to_true_file_name,
# 		'With to Without file name': true_to_false_file_name
# 	}
# 	json.dump(data, j, ensure_ascii=False, indent=2)

