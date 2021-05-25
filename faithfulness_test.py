import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import copy

from grad_aam import GradCam, preprocess_image
from model import ResNet, BasicBlock
from utils import classes

image = './data/train1/Images/pant_length_labels/004c0a244b1c695f7e37ef030dd97955.jpg'
trans = transforms.CenterCrop((100, 512))
loader = transforms.ToTensor()
resize = transforms.Resize((512, 512))

img = Image.open(image)
img = trans(img)
# img.save('D:/毕业论文/论文材料/裁剪图.jpg', format='JPG')

LOAD_PATH = './models/model_sleeve_length_thresh_0.6_v2.pth'

device = torch.device('cuda')
resnet = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], attribute=classes[5]).to(device)
resnet.load_state_dict(torch.load(LOAD_PATH))
criterion = nn.BCELoss()
m = nn.Sigmoid()

img_tensor = loader(resize(img)).unsqueeze(0).to(device)
label = torch.tensor([[0., 0., 0., 0., 1., 0.]], dtype=torch.float32).to(device)
resnet.eval()
output = resnet(img_tensor).to(torch.float32)
loss = criterion(m(output), label)
print(loss)


"""
def show_grad_aam(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    cv2.imwrite('D:/heatmap.jpg', heatmap)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    cv2.imwrite('D:/heatmap-CAM.jpg', np.uint8(255 * cam))


model = copy.deepcopy(resnet)
del model.fc1_coat_length
del model.fc2_collar_design
del model.fc3_lapel_design
del model.fc4_neck_design
del model.fc5_neckline_design
del model.fc6_pant_length
del model.fc7_skirt_length
del model.fc8_sleeve_length

grad_cam = GradCam(model, target_layer_names=['layer4'])

img = np.array(img)
img = np.float32(cv2.resize(img, (512, 512))) / 255
input = preprocess_image(img)
input.requires_grad = True
target_index = None
mask = grad_cam(input, target_index)
show_grad_aam(img, mask)
"""
