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
import torch
from pysaliency.plotting import visualize_distribution
from torchvision import transforms, utils, models
import torch.nn as nn
from utils.data_process import preprocess_img, postprocess_img

# command will be like: 
# python model.py --image "../../results_sd15/generated_image" --mask "../../results_sd15/mask" --output "../predicted_hole/VGG19"
ap = argparse.ArgumentParser()
ap.add_argument('--image', help='path of image')
ap.add_argument('--mask', help='path of mask')
ap.add_argument('--output', help='output dir name')
ap.add_argument('--debug', action='store_true', help='Enable debug mode')
args = ap.parse_args()

device = torch.device("cpu")
w = 512
h = 512

def ShowImage(im, title='', ax=None):
  if ax is None:
    P.figure()
  P.axis('off')
  P.imshow(im)
  P.title(title)
  P.show()

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

def normalize(image, region): # make sure the sum of score within region have same value
    region = np.where(region == 1, 1, 0)
    matrix = image * region
    image = image / np.sum(matrix) * np.sum(region)
    #image = (image - image.min()) / (image.max() - image.min())
    #image = image * w * h * 0.5 / np.sum(image)
    return image

def CalASVS(saliencyMap, region):
    # remove unreasonable region
    region = np.where(region == 1, 1, 0)
    # for i in range(len(region)):
    #     for j in range(len(region[0])):
    #         region[i][j] = 1 if region[i][j] == 1 else 0
    matrix = saliencyMap * region
    area = cv2.countNonZero(region)
    return pow(LA.norm(matrix), 2) / area

def CalGD(saliencyMap, region):
    # remove unreasonable region
    region = np.where(region == 1, 1, 0)
    # for i in range(len(region)):
    #     for j in range(len(region[0])):
    #         region[i][j] = 1 if region[i][j] == 1 else 0
    matrix = saliencyMap * region
    area = cv2.countNonZero(region)
    return np.sum(matrix) / area

def LoadTranSalNet():
    flag = 1 # 0 for TranSalNet_Dense, 1 for TranSalNet_Res

    if flag:
        from TranSalNet_Res import TranSalNet
        model = TranSalNet()
        model.load_state_dict(torch.load(r'pretrained_models/TranSalNet_Res.pth', map_location=torch.device('cpu')))
    else:
        from TranSalNet_Dense import TranSalNet
        model = TranSalNet()
        model.load_state_dict(torch.load(r'pretrained_models/TranSalNet_Dense.pth', map_location=torch.device('cpu')))

    model = model.to(device) 
    model.eval()
    return model


def TranSalNetModel(im_orig):
    image = preprocess_img(im_orig) # padding and resizing input image into 384x288
    image = np.array(image)/255.
    image = np.expand_dims(np.transpose(image,(2,0,1)),axis=0)
    image = torch.from_numpy(image)
    image = image.type(torch.FloatTensor).to(device)
    pred_saliency = model(image)
    toPIL = transforms.ToPILImage()
    pic = toPIL(pred_saliency.squeeze())

    pred_saliency = postprocess_img(pic) # restore the image to its original size as the result
    # f, axs = P.subplots(nrows=1, ncols=2, figsize=(12, 2))
    # axs[0].imshow(im_orig)
    # axs[0].set_axis_off()
    # axs[1].matshow(pred_saliency)  # first image in batch, first (and only) channel
    # axs[1].set_axis_off()
    return pred_saliency


def GetMask(mask, DiEr=False):
    #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)   # revert color(0 to 1, 1 to 0)

    # try to capture hole which is not entirely closed
    mask_copy = mask.copy()
    if DiEr:
        kernel = np.ones((20, 20), np.uint8)
        mask_copy = cv2.dilate(mask_copy, kernel, iterations=1)
        mask_copy = cv2.erode(mask_copy, kernel, iterations=1)

    # fill the pixel inside the border
    def FillMask(mask):
        mask_copy = mask.copy()
        mask_copy[0, :] = 0  # Set the first row to 0
        mask_copy[-1, :] = 0  # Set the last row to 0
        mask_copy[:, 0] = 0  # Set the first column to 0
        mask_copy[:, -1] = 0  # Set the last column to 0
        
        # Perform flood fill starting from the outside point
        cv2.floodFill(mask_copy, None, (0, 0), 2)

        mask_copy = np.where(mask_copy == 2, 0, 1)

        return mask_copy.astype(np.uint8)

    # extend the mask
    def ExtendMaskBorder(mask):
        # Define a kernel for dilation
        kernel = np.ones((int(w / 10), int(h / 10)), np.uint8)

        # Perform dilation on the mask
        extended_mask = cv2.dilate(mask, kernel, 1)

        return extended_mask.astype(np.uint8)

    filled_mask = FillMask(mask_copy)
    extended_mask = ExtendMaskBorder(mask)

    return mask, filled_mask, extended_mask

def GetProduct(image, mask):
    product = cv2.bitwise_and(image, image, mask=mask)
    return product

def GetBorder(mask):
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


def HoleIssueDetection(image_path, mask_path):
    image = LoadImage(image_path)
    mask = cv2.resize(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE), (w, h))
    mask = np.where(mask == 0, 1, 0)
    mask = mask.astype(np.uint8)
    omask, fmask, emask = GetMask(mask)
    # check whether contains hole
    hole = (fmask - omask).astype("uint8")
    hole = np.where(hole != 1, 0, hole)
    area = np.sum(hole)
    if area < w * h / 400:  # without hole
        omask, fmask, emask = GetMask(mask, DiEr=True)
        # check whether contains hole
        hole = (fmask - omask).astype("uint8")
        hole = np.where(hole != 1, 0, hole)
        area = np.sum(hole)
        #print('enclose area: ', area)
        if area < w * h / 100:
            return 0, 0, 0, 0, 0, 0, 0, 0

    # denoise the hole
    kernel = np.ones((10, 10), np.uint8)
    hole = cv2.erode(hole, kernel, iterations=1)
    hole = cv2.dilate(hole, kernel, iterations=1)

    hole_border = GetBorder(hole)
    kernel = np.ones((10, 10), np.uint8)
    hole_border = cv2.dilate(hole_border, kernel, iterations=1)

    saliencyMap = normalize(TranSalNetModel(image), mask)

    border = (emask - fmask).astype("uint8")
    border = np.where(border != 1, 0, border)


    kernel = np.ones((20, 20), np.uint8)
    extend_hole = cv2.dilate(hole, kernel, iterations=1)
    extend_hole_border = extend_hole - hole

    product = GetProduct(image, mask)
    productMap = normalize(TranSalNetModel(product), mask)

    

    # f, axs = P.subplots(nrows=1, ncols=4, figsize=(12, 4))
    # axs[0].imshow(image)
    # axs[0].set_axis_off()
    # axs[1].matshow(saliencyMap)  # first image in batch, first (and only) channel
    # axs[1].set_axis_off()
    # axs[2].imshow(product)
    # axs[2].set_axis_off()
    # axs[3].matshow(productMap)  # first image in batch, first (and only) channel
    # axs[3].set_axis_off()

    hole_after_ASVS = CalASVS(saliencyMap, extend_hole)
    product_after_ASVS = CalASVS(saliencyMap, mask)
    hole_before_ASVS = CalASVS(productMap, extend_hole)
    product_before_ASVS = CalASVS(productMap, mask)
    hole_after_GD = CalGD(saliencyMap, extend_hole)
    product_after_GD = CalGD(saliencyMap, mask)
    hole_before_GD = CalGD(productMap, extend_hole)
    product_before_GD = CalGD(productMap, mask)

    return hole_after_ASVS, product_after_ASVS, hole_before_ASVS, product_before_ASVS, hole_after_GD, product_after_GD, hole_before_GD, product_before_GD

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
            hole_after_ASVS, product_after_ASVS, hole_before_ASVS, product_before_ASVS, hole_after_GD, product_after_GD, hole_before_GD, product_before_GD = HoleIssueDetection(f'{image}/{file_name}', f'{mask}/{file_name.split("_")[0]}.png')
            break
        with open(f'{args.output}/{file_name}.json', 'w') as f:
            #print(f'{image}/{file_name}', f'{mask}/{file_name}')
            hole_after_ASVS, product_after_ASVS, hole_before_ASVS, product_before_ASVS, hole_after_GD, product_after_GD, hole_before_GD, product_before_GD = HoleIssueDetection(f'{image}/{file_name}', f'{mask}/{file_name.split("_")[0]}.png')
            data = {
                "hole_after_ASVS": hole_after_ASVS,
                "product_after_ASVS": product_after_ASVS,
                "hole_before_ASVS": hole_before_ASVS,
                "product_before_ASVS": product_before_ASVS,
                "hole_after_GD": hole_after_GD,
                "product_after_GD": product_after_GD,
                "hole_before_GD": hole_before_GD,
                "product_before_GD": product_before_GD

            }
            json.dump(data, f, ensure_ascii=False, indent=2)
        
    #P.show()

model = LoadTranSalNet()
main()
