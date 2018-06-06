from __future__ import division
import os, sys, time
import _pickle
import itertools
sys.path.append('/Users/Blair/cocoapi/PythonAPI')

# matplotlib inline
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (8.0, 10.0)

import cv2
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('jpg')
from PIL import Image, ImageDraw

import torch
import torch.nn as nn

from math import ceil, floor


class DataLoader:
    def __init__(self, dataDir, dataType):
        self.dataDir = dataDir
        self.dataType = dataType
        self.id = []
        self.feats = []
        self.label = []

    def get_bboxes(self, img, ss, num_rects=2000):
        try:
            ss.setBaseImage(img)
            # ss.switchToSelectiveSearchQuality() # good quality search
            ss.switchToSelectiveSearchFast()  # fast search
            # t1 = time.time()
            rects = ss.process()
            # t1 = time.time() - t1
            return rects[:num_rects]
        except KeyboardInterrupt:
            print('keyboard interrupt')
            sys.exit()
        except:
            return None

    # IoU
    def iou(self, rect1, rect2):  # rect = [x, y, w, h]
        x1, y1, w1, h1 = rect1
        X1, Y1 = x1 + w1, y1 + h1
        x2, y2, w2, h2 = rect2
        X2, Y2 = x2 + w2, y2 + h2
        a1 = (X1 - x1 + 1) * (Y1 - y1 + 1)
        a2 = (X2 - x2 + 1) * (Y2 - y2 + 1)
        x_int = max(x1, x2)
        X_int = min(X1, X2)
        y_int = max(y1, y2)
        Y_int = min(Y1, Y2)
        a_int = (X_int - x_int + 1) * (Y_int - y_int + 1) * 1.0
        if x_int > X_int or y_int > Y_int:
            a_int = 0.0
        return a_int / (a1 + a2 - a_int)

    # nearest neighbor in 1-based indexing
    def _nnb_1(self, x):
        x1 = int(floor((x + 8) / 16.0))
        x1 = max(1, min(x1, 13))
        return x1

    def project_onto_feature_space(self, rect, image_dims):
        # project bounding box onto conv net
        # @param rect: (x, y, w, h)
        # @param image_dims: (imgx, imgy), the size of the image
        # output bbox: (x, y, x'+1, y'+1) where the box is x:x', y:y'

        # For conv 5, center of receptive field of i is i_0 = 16 i for 1-based indexing
        imgx, imgy = image_dims
        x, y, w, h = rect
        # scale to 224 x 224, standard input size.
        x1, y1 = ceil((x + w) * 224 / imgx), ceil((y + h) * 224 / imgy)
        x, y = floor(x * 224 / imgx), floor(y * 224 / imgy)
        px = self._nnb_1(x + 1) - 1  # inclusive
        py = self._nnb_1(y + 1) - 1  # inclusive
        px1 = self._nnb_1(x1 + 1)  # exclusive
        py1 = self._nnb_1(y1 + 1)  # exclusive

        return [px, py, px1, py1]

    class Featurizer:
        dim = 11776  # for small features

        def __init__(self):
            # pyramidal pooling of sizes 1, 3, 6
            self.pool1 = nn.AdaptiveMaxPool2d(1)
            self.pool3 = nn.AdaptiveMaxPool2d(3)
            self.pool6 = nn.AdaptiveMaxPool2d(6)
            self.lst = [self.pool1, self.pool3, self.pool6]

        def featurize(self, projected_bbox, image_features):
            # projected_bbox: bbox projected onto final layer
            # image_features: C x W x H tensor : output of conv net
            full_image_features = torch.from_numpy(image_features)
            x, y, x1, y1 = projected_bbox
            crop = full_image_features[:, x:x1, y:y1]
            #         return torch.cat([self.pool1(crop).view(-1), self.pool3(crop).view(-1),
            #                           self.pool6(crop).view(-1)], dim=0) # returns torch Variable
            return torch.cat([self.pool1(crop).view(-1), self.pool3(crop).view(-1),
                              self.pool6(crop).view(-1)], dim=0).data.numpy()  # returns numpy array

    def load(self):
        t1 = time.time()
        annFile = '{}/annotations/instances_{}.json'.format(self.dataDir, self.dataType)  # annotations

        # initialize COCO api for instance annotations. This step takes several seconds each time.
        coco = COCO(annFile)

        cats = coco.loadCats(coco.getCatIds())  # categories
        cat_id_to_name = {cat['id']: cat['name'] for cat in cats}  # category id to name mapping
        cat_name_to_id = {cat['name']: cat['id'] for cat in cats}  # category name to id mapping

        cat_to_supercat = {cat['name']: cat['supercategory'] for cat in cats}
        cat_id_to_supercat = {cat['id']: cat['supercategory'] for cat in cats}

        cat = ["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
               "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat"]
        true_ids = [cat_name_to_id[c] for c in cat]

        # read features:
        [img_list, bboxes] = _pickle.load(open(os.path.join(self.dataDir, 'bboxes_retrieval',
                                                            '{}_bboxes_retrieval.p'.format(self.dataType)), 'rb'),
                                          encoding='latin1')

        [img_ids, feats] = _pickle.load(open(os.path.join(self.dataDir, 'features2_small', '{}.p'.format(self.dataType)), 'rb'),
                                        encoding='latin1')

        n = len(img_ids)

        for i in range(n):
            if bboxes[i] is None:  ### OpenCV has thrown an error. Discard image.
                print('Discard image from consideration.')
                continue
            # print('# bboxes per img: ', len(bboxes[i]))
            if self.dataType == "val2014" or self.dataType == "test2014":
                bboxes[i] = random.choice(bboxes[i])

            img_id = img_ids[i]
            img = coco.loadImgs([img_id])[0]
            annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=None)
            anns = coco.loadAnns(annIds)

            categories = set([ann['category_id'] for ann in anns])

            for bbox in bboxes[i]:
                lab = np.zeros(19)
                found = False
                for index in range(len(true_ids)):
                    true_id = true_ids[index]
                    if categories.__contains__(true_id) is False:
                        continue
                    for ann in anns:
                        if ann['category_id'] == true_id and self.iou(bbox, ann['bbox']) > 0.5:
                            found = True
                            lab[index] = 1
                            break
                    if found:
                        break
                if found is False:
                    lab[18] = 1

                self.label.append(lab)

                img_feats = feats[i]
                dataType = '{}_2'.format(self.dataType)
                img_pil = Image.open(
                    '%s/%s/%s' % (self.dataDir, dataType, img['file_name']))  # make sure data dir is correct

                featurizer = self.Featurizer()
                projected_bbox = self.project_onto_feature_space(bbox, img_pil.size)
                bbox_feats = featurizer.featurize(projected_bbox, img_feats)
                self.feats.append(bbox_feats.flatten())
                if dataType == "train2014":
                    self.id.append(img_id)

        self.feats = np.array(self.feats)
        self.label = np.array(self.label)
        self.id = np.array(self.id)
        print(self.dataType, ":", self.feats.shape)
        print(self.dataType, ":", self.label.shape)
        print(self.dataType, ":", self.id.shape)
        t2 = time.time()
        print("%d seconds, data pre-processing done" % (t2 - t1))
