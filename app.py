import os
from flask import Flask, render_template, request, redirect, url_for
import json
import pandas as pd

app = Flask(__name__)

IMAGE_DIR = os.path.join('static', 'generated_image')
REAL_IMAGE_DIR = os.path.join('static', 'real_image')
MASK_IMAGE_DIR = os.path.join('static', 'mask')
LABEL_DIR = os.path.join('static', 'labels')
CSV_FILE = os.path.join('static', 'prompt.csv')

os.makedirs(LABEL_DIR, exist_ok=True)

def dir_files(img_dir):
    return [os.path.join(img_dir, filename) for filename in os.listdir(img_dir) 
            if filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.json'))]

images = dir_files(IMAGE_DIR)
images.sort()

real_images = dir_files(REAL_IMAGE_DIR)
real_images.sort()

prompt_csv = pd.read_csv(CSV_FILE)
prompt_csv['generated_image'] = prompt_csv['generated_image'].apply(lambda x: "/".join(x.split("/")[-2:]))

mask_images = dir_files(MASK_IMAGE_DIR)
mask_images.sort()

if len(images) != len(real_images) or len(images) != len(mask_images):
    print(f"Number of file in {REAL_IMAGE_DIR} {IMAGE_DIR} and {MASK_IMAGE_DIR} are not equal! exit!")
    exit(1)


@app.route('/')
def index():
    currentImageIndex= 0

    if 'currentImageIndex' in request.args:
        currentImageIndex = int(request.args['currentImageIndex'])
    
    if currentImageIndex >= len(images):
        return render_template('message.html', message="Congrats! You have finished labeling all the images! Rest Well!")

    labelFiles = list(map(lambda x: x.split("/")[-1], dir_files(LABEL_DIR)))

    labelFileName = images[currentImageIndex].split("/")[-1].split(".")[0] + '.json'

    prompt = prompt_csv.loc[prompt_csv['generated_image'] == "/".join(images[currentImageIndex].split("/")[-2:])]['prompt'].values

    return render_template('index.html', image=images[currentImageIndex], 
                           real_image=real_images[currentImageIndex], 
                           mask_image=mask_images[currentImageIndex], 
                           current_image_index = currentImageIndex,
                           prompt = prompt[0] if len(prompt) else "No Prompt!",
                           total = len(real_images),
                           labeled = labelFileName in labelFiles)


@app.route('/submit', methods=['POST'])
def submit():    
    selected_options = {}
    currentImageIndex = int(request.form.get('current_image_index'))

    for option in ['weird_shadow', 'product_extend', 'product_hole_issue', 'floating_product', 
                   'background_prompt_not_match', 'illogical']:
        selected_options[option] = request.form.get(option) is not None
    
    for option in ['aesthetic_score']:
        selected_options[option] = request.form.get(option)
    
    current_image_filename = images[currentImageIndex]
    json_filename = os.path.splitext(os.path.basename(current_image_filename))[0] + '.json'

    with open(os.path.join(LABEL_DIR, json_filename), 'w') as json_file:
        json.dump(selected_options, json_file)

    currentImageIndex += 1

    return redirect(url_for('index', currentImageIndex=currentImageIndex))


@app.route('/navigation', methods=['POST'])
def navigation():
    currentImageIndex = int(request.form.get('current_image_index'))
    if "previous" in request.form:
        currentImageIndex -= 1
        currentImageIndex = max(0, currentImageIndex)
    elif "next" in request.form:
        currentImageIndex += 1
        currentImageIndex = min(len(images)-1, currentImageIndex)
    elif "latest-labeled" in request.form:
        labelFiles = list(map(lambda x: x.split("/")[-1], dir_files(LABEL_DIR)))

        if len(labelFiles) == len(images):
            return render_template('message.html', message="Congrats! You have finished labeling all the images! Rest Well!")

        labelFilesWOExt = set(map(lambda x: x.split("/")[-1].split('.')[0], labelFiles))
        imageFilesWOExt = set(map(lambda x: x.split("/")[-1].split('.')[0], images))

        unlabeledIndex = list(imageFilesWOExt - labelFilesWOExt)
        unlabeledIndex.sort()

        imageFilesWOExt = list(imageFilesWOExt)
        imageFilesWOExt.sort()
        index = list(imageFilesWOExt).index(unlabeledIndex[0]) if len(unlabeledIndex) else len(images)
        currentImageIndex = max(index-1, 0)

    return redirect(url_for('index', currentImageIndex=currentImageIndex))


if __name__ == '__main__':
    app.run(debug=True)
