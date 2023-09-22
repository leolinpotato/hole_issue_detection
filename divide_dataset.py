import pandas as pd
import os

options = [1, 2, 3, 4, 5, 6]

# PLEASE CHANGE THIS ACCORDING TO YOUR ID
option = 1

if option not in options:
    print("Option not identified")
    exit()

IMAGE_DIR = os.path.join('static', 'generated_image')
REAL_IMAGE_DIR = os.path.join('static', 'real_image')
MASK_IMAGE_DIR = os.path.join('static', 'mask')
CSV_FILE = os.path.join('static', 'prompt.csv')

if not os.path.exists(CSV_FILE):
    print(f"prompt.csv does not exist! Please put prompt.csv in {CSV_FILE}")

df = pd.read_csv(CSV_FILE)

# 2040 --> 6 --> 340 images
images_per_member = int(len(df) / len(options))

images_range_start = (option-1) * images_per_member

def dir_files(img_dir):
    return [os.path.join(img_dir, filename) for filename in os.listdir(img_dir) 
            if filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.json'))]

images_files = dir_files(IMAGE_DIR)
real_images_files = dir_files(REAL_IMAGE_DIR)
mask_files = dir_files(MASK_IMAGE_DIR)

images_files.sort()
real_images_files.sort()
mask_files.sort()

print("Deleting files, please wait a moment!")

# Delete file that doesn't belong to the person
for i in range(images_range_start):
    os.remove(images_files[i])
    os.remove(real_images_files[i])
    os.remove(mask_files[i])

for i in range(images_range_start+images_per_member, len(images_files)):
    os.remove(images_files[i])
    os.remove(real_images_files[i])
    os.remove(mask_files[i])

print("Finish! Data ready to be labeled!")
