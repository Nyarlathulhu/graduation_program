import torch
import cv2
import numpy as np
import os
import copy

from model import ResNet, BasicBlock
from utils import classes

LOAD_PATH = './models/model_sleeve_length_thresh_0.6_v2.pth'

device = torch.device('cuda')
resnet = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], attribute=classes[7]).to(device)
resnet.load_state_dict(torch.load(LOAD_PATH))


class FeatureExtractor:
    """
        class for extracting activations and registering gradients from target
        intermediate layers
    """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]

        return outputs, x


class ModelOutputs:
    """
        class for making a forward pass, and getting:
            1. network output
            2. activations from intermediate target layers
            3. gradients from intermediate target layers
    """

    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        if self.model.attr == 'coat_length':
            output = resnet.fc1_coat_length(output)
        elif self.model.attr == 'collar_design':
            output = resnet.fc2_collar_design(output)
        elif self.model.attr == 'lapel_design':
            output = resnet.fc3_lapel_design(output)
        elif self.model.attr == 'neck_design':
            output = resnet.fc4_neck_design(output)
        elif self.model.attr == 'neckline_design':
            output = resnet.fc5_neckline_design(output)
        elif self.model.attr == 'pant_length':
            output = resnet.fc6_pant_length(output)
        elif self.model.attr == 'skirt_length':
            output = resnet.fc7_skirt_length(output)
        elif self.model.attr == 'sleeve_length':
            output = resnet.fc8_sleeve_length(output)

        return target_activations, output


class GradCam:

    def __init__(self, model, target_layer_names):
        self.model = model.cuda().eval()
        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        features, output = self.extractor(input.cuda())
        # feature size [1, 512, 16, 16]
        if index is None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.Tensor(torch.from_numpy(one_hot))
        one_hot.requires_grad = True
        one_hot = torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (512, 512))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


def preprocess_image(img):
    means = [0.5, 0.5, 0.5]
    stds = [0.5, 0.5, 0.5]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)

    input = preprocessed_img
    input.requires_grad = True

    return input


def show_grad_aam(img, mask, name):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    cv2.imwrite('./feature_map/train2/sleeve_length/{}'.format(name), heatmap)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite('./grad_aam_image/train2/sleeve_length/{}'.format(name), np.uint8(255 * cam))


model = copy.deepcopy(resnet)
del model.fc1_coat_length
del model.fc2_collar_design
del model.fc3_lapel_design
del model.fc4_neck_design
del model.fc5_neckline_design
del model.fc6_pant_length
del model.fc7_skirt_length
del model.fc8_sleeve_length

grad_aam = GradCam(model, target_layer_names=['layer4'])

bound = 20
IMG_PATH = './data/train2/Images/sleeve_length_labels/'
image = []
i = 0
print("target images:")
for root, dirs, filename in os.walk(IMG_PATH):
    print(filename)
for f in filename[:bound]:
    image.append(cv2.imread(IMG_PATH + f, 1))
for img in image:
    img = np.float32(cv2.resize(img, (512, 512))) / 255
    input = preprocess_image(img)
    input.requires_grad = True
    # print("input.size() = ", input.size())

    target_index = None
    mask = grad_aam(input, target_index)
    show_grad_aam(img, mask, filename[i])
    i += 1
