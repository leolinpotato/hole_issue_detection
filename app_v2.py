import os
from flask import Flask, render_template, request, redirect, url_for, send_file
import json
import pandas as pd
import zipfile


app = Flask(__name__)

IMAGE_DIR = os.path.join('static', 'generated_image')
LABEL_DIR = os.path.join('static', 'bg_labels')

os.makedirs(LABEL_DIR, exist_ok=True)


def dir_files(img_dir):
    return [os.path.join(img_dir, filename) for filename in os.listdir(img_dir) 
            if filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.json'))]


images = dir_files(IMAGE_DIR)
images.sort()


@app.route('/')
def index():
    currentImageIndex= 0

    if 'currentImageIndex' in request.args:
        currentImageIndex = int(request.args['currentImageIndex'])
    
    if currentImageIndex >= len(images):
        return render_template('message.html', message="Congrats! You have finished labeling all the images! Rest Well!")

    labelFiles = list(map(lambda x: x.split("/")[-1], dir_files(LABEL_DIR)))

    labelFileName = images[currentImageIndex].split("/")[-1].split(".")[0] + '.json'

    current_image_filename = images[currentImageIndex]
    json_filename = os.path.splitext(os.path.basename(current_image_filename))[0] + '.json'

    labeled_data = {}
    if labelFileName in labelFiles:
        f = open(f"{LABEL_DIR}/{json_filename}")
        labeled_data = json.load(f)

    return render_template('index_v2.html', image=images[currentImageIndex], 
                           current_image_index = currentImageIndex,
                           prompt = "No Prompt!",
                           total = len(images),
                           labeled = labelFileName in labelFiles, 
                           labeled_data = labeled_data)


@app.route('/submit', methods=['POST'])
def submit():    
    selected_options = {}
    currentImageIndex = int(request.form.get('current_image_index'))

    for option in ['product_hole_issue']:
        selected_options[option] = request.form.get(option) is not None
    
    for option in ['realistic_score', 'aesthetic_score']:
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
    elif "first-image" in request.form:
        currentImageIndex = 0
    elif "last-image" in request.form:
        currentImageIndex = len(images) - 1
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


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), 
                       os.path.relpath(os.path.join(root, file), 
                                       os.path.join(path, '..')))


@app.route('/download-json', methods=['POST'])
def downloadJson():
    with zipfile.ZipFile('data.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipdir('static/bg_labels/', zipf)

    return send_file('data.zip', as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
