import numpy as np
import PIL.Image
import cv2
from numpy import linalg as LA
import torch
from torchvision import transforms, utils, models
import torch.nn as nn
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'TranSalNet'))

from utils.data_process import preprocess_img, postprocess_img
from TranSalNet_Res import TranSalNet

class ProductHoleIssue():
    def __init__(self, device) -> None:
        model = TranSalNet()
        model.load_state_dict(torch.load(r'TranSalNet/pretrained_models/TranSalNet_Res.pth', map_location=device))
        model = model.to(device)
        model.eval()
        self.model = model
        self.device = device
        self.w = 512
        self.h = 512

    def load_image(self, file_path):
      im = PIL.Image.open(file_path)
      im = im.resize((self.w, self.h))
      im = np.asarray(im)
      if im.dtype != np.uint8:
        im = im.astype(np.uint8)
      return im

    def normalize(self, image, region): # make sure the sum of score within region have same value
        region = np.where(region == 1, 1, 0)
        matrix = image * region
        image = image / np.sum(matrix) * np.sum(region)
        return image

    def calculate_ASVS(self, saliencyMap, region):
        # remove unreasonable region
        region = np.where(region == 1, 1, 0)
        matrix = saliencyMap * region
        area = cv2.countNonZero(region)
        return pow(LA.norm(matrix), 2) / area

    def TranSalNetModel(self, im_orig):
        image = preprocess_img(im_orig) # padding and resizing input image into 384x288
        image = np.array(image)/255.
        image = np.expand_dims(np.transpose(image,(2,0,1)),axis=0)
        image = torch.from_numpy(image)
        image = image.type(torch.FloatTensor).to(self.device)
        pred_saliency = self.model(image)
        toPIL = transforms.ToPILImage()
        pic = toPIL(pred_saliency.squeeze())

        pred_saliency = postprocess_img(pic) # restore the image to its original size as the result
        return pred_saliency


    def get_mask(self, mask, DiEr=False):
        # try to capture hole which is not entirely closed
        mask_copy = mask.copy()
        if DiEr:
            kernel = np.ones((20, 20), np.uint8)
            mask_copy = cv2.dilate(mask_copy, kernel, iterations=1)
            mask_copy = cv2.erode(mask_copy, kernel, iterations=1)

        # fill the pixel inside the border
        def fill_mask(mask):
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
        def extend_mask_border(mask):
            # Define a kernel for dilation
            kernel = np.ones((int(self.w / 10), int(self.h / 10)), np.uint8)

            # Perform dilation on the mask
            extended_mask = cv2.dilate(mask, kernel, 1)

            return extended_mask.astype(np.uint8)

        filled_mask = fill_mask(mask_copy)
        extended_mask = extend_mask_border(mask)

        return mask, filled_mask, extended_mask

    def get_product(self, image, mask):
        product = cv2.bitwise_and(image, image, mask=mask)
        return product

    def detect(self, image_path, mask_path):
        image = self.load_image(image_path)
        mask = cv2.resize(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE), (self.w, self.h))
        mask = np.where(mask == 0, 1, 0)
        mask = mask.astype(np.uint8)
        omask, fmask, emask = self.get_mask(mask)
        # check whether contains hole
        hole = (fmask - omask).astype("uint8")
        hole = np.where(hole != 1, 0, hole)
        area = np.sum(hole)
        if area < self.w * self.h / 400:  # without hole
            omask, fmask, emask = self.get_mask(mask, DiEr=True)
            # check whether contains hole
            hole = (fmask - omask).astype("uint8")
            hole = np.where(hole != 1, 0, hole)
            area = np.sum(hole)
            #print('enclose area: ', area)
            if area < self.w * self.h / 100:
                return False

        # denoise the hole
        kernel = np.ones((10, 10), np.uint8)
        hole = cv2.erode(hole, kernel, iterations=1)
        hole = cv2.dilate(hole, kernel, iterations=1)

        saliencyMap = self.normalize(self.TranSalNetModel(image), mask)

        kernel = np.ones((20, 20), np.uint8)
        extend_hole = cv2.dilate(hole, kernel, iterations=1)

        product = self.get_product(image, mask)
        productMap = self.normalize(self.TranSalNetModel(product), mask)

        hole_after_ASVS = self.calculate_ASVS(saliencyMap, extend_hole)
        product_after_ASVS = self.calculate_ASVS(saliencyMap, mask)
        hole_before_ASVS = self.calculate_ASVS(productMap, extend_hole)
        product_before_ASVS = self.calculate_ASVS(productMap, mask)

        r = (hole_after_ASVS / product_after_ASVS) / (hole_before_ASVS / product_before_ASVS)
        return r > 0.8
            
if __name__ == "__main__":
    device = "cpu"
    product_hole_issue_detector = ProductHoleIssue(torch.device(device))
    img_path = ''
    ori_mask_path = ''

    print(product_hole_issue_detector.detect(img_path, ori_mask_path))