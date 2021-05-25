import torch
import torch.nn as nn
import torch.optim as optim
import warnings

from model import BasicBlock, ResNet
from utils import train1_dataloader, train2_dataloader, classes

warnings.filterwarnings('ignore')

EPOCH = 10
LR = 0.01
SAVE_PATH = './models/model_sleeve_length_thresh_0.6_v2.pth'
LOAD_PATH = './models/model_sleeve_length_thresh_0.6.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# set model(resnet18) for each label
model = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], attribute=classes[7]).to(device)
model.load_state_dict(torch.load(LOAD_PATH))
criterion = nn.BCELoss()
m = nn.Sigmoid()
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=0.0005)


def calc_P_R(x, y, thresh=0.5):
    pred = x > thresh
    pred = pred.float()
    # predicted num of labels that one image has
    pred_label_num = torch.sum(pred)
    if pred_label_num == 0:
        return 0, 0
    # true num of labels that one image has
    true_label_num = torch.sum(y)
    # num of correct predicted labels
    true_pred_num = torch.sum(pred * y)
    precision_ = true_pred_num / pred_label_num
    recall_ = true_pred_num / true_label_num

    return precision_.item(), recall_.item()


for epoch in range(EPOCH):
    model.train()
    loss_sum = 0.0
    running_precision = 0.0
    running_recall = 0.0

    # training
    for i, data in enumerate(train2_dataloader, 0):
        length = len(train2_dataloader)
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        output = outputs.to(torch.float32)
        label = labels.to(torch.float32)
        loss = criterion(m(output), label)
        precision, recall = calc_P_R(m(output), label, thresh=0.6)
        running_precision += precision
        running_recall += recall

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        P = running_precision / (i + 1)
        R = running_recall / (i + 1)
        print('[epoch:%d, iter:%d] Loss: %.4f | P: %.4f \t R: %.4f'
              % (epoch + 1, (i + 1 + epoch * length), loss_sum / (i + 1),
                 P, R))

    # testing
    print('Testing...')
    with torch.no_grad():
        loss_sum = 0.0
        running_precision = 0.0
        running_recall = 0.0
        for data in train1_dataloader:
            model.eval()
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            output = outputs.to(torch.float32)
            label = labels.to(torch.float32)
            loss = criterion(m(output), label)
            precision, recall = calc_P_R(m(output), label, thresh=0.6)
            running_precision += precision
            running_recall += recall
            loss_sum += loss.item()
        total = train1_dataloader.__len__()
        P = running_precision / total
        R = running_recall / total
        # F1 = 2 * P * R / (P + R)
        print('Test Loss: %.4f \t P: %.4f \t R: %.4f' %
              (loss_sum / total, P, R))

    torch.save(model.state_dict(), SAVE_PATH)
    print('Model saved.')

print('Training completed.')
