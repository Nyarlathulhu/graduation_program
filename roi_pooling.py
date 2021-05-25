import cv2
import torch
import numpy as np
from torchvision import transforms
from torchvision.ops import roi_pool

import os
import json

# from model import num_classes


FEAT_PATH = './grad_aam_image/train2/sleeve_length/'
ROI_PATH = './results/roi_pos/train2/sleeve_length/'
features = []
attr_repres = {}
i = 0
trans = transforms.ToTensor()

for root, dirs, feats in os.walk(FEAT_PATH):
    print(feats)
for f in feats:
    features.append(cv2.imread(FEAT_PATH + f, 1))

for feature in features:
    feat_tensor = trans(feature).unsqueeze(0)
    rois = np.load(ROI_PATH + feats[i] + '.npy')
    rois = torch.from_numpy(rois)
    rois[2] += rois[0]
    rois[3] += rois[1]
    box = torch.tensor([[0, rois[0], rois[1], rois[2], rois[3]]]).float()
    attr_representation = roi_pool(feat_tensor, box, output_size=(7, 7))
    name = feats[i]
    attr_repres[name] = attr_representation.tolist()
    i += 1
    # print(attr_representation.size())

json_file = json.dumps(attr_repres)
with open('./results/attribute_representations/train2/sleeve_length.json', 'w') as file:
    file.write(json_file)
