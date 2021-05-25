import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import pandas as pd
from PIL import Image
# import numpy as np
# import cv2


classes = ['coat_length', 'collar_design', 'lapel_design', 'neck_design',
           'neckline_design', 'pant_length', 'skirt_length', 'sleeve_length']
BATCH_SIZE = 20


# define dataset
class ImgDataset(Dataset):
    def __init__(self, label_file, trans=None, train=True):
        """
        :param label_file: contain labels info, format: csv
        :param trans: transforms
        :param train: train set or test set
        """
        self.file = pd.read_csv(label_file)
        for i in range(len(self.file)):
            self.file['label'][i] = self.file['label'][i].replace('tag', '')
        self.paths = self.file['img_path'].tolist()
        self.labels = self.file['label'].tolist()
        self.transform = trans
        self.is_train = train

    def __getitem__(self, item):
        if self.is_train:
            # img_path = './data/train1/' + self.paths[item]
            img_path = './data/train2/' + self.paths[item]
        else:
            img_path = './data/train1/' + self.paths[item]
            # img_path = './data/train2/' + self.paths[item]
        img_id = img_path.split('/')[-1]
        img = Image.open(img_path)
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            img = self.transform(img)
        label = list(map(int, self.labels[item]))
        # label_original = list(self.labels[item])
        # label = []
        # for j in range(len(label_original)):
        #     label.append(int(label_original[j]))

        return torch.tensor(img).float(), torch.tensor(label)

    def __len__(self):
        return len(self.paths)


LABEL = classes[7]

data_transform = {
    'train': transforms.Compose([
        # transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        # mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
}

# train1 = ImgDataset(label_file='./data/train1_'+LABEL+'_labels.csv',
#                     trans=data_transform['train'], train=True)
train1 = ImgDataset(label_file='./data/train1_'+LABEL+'_labels.csv',
                    trans=data_transform['train'], train=False)
train2 = ImgDataset(label_file='./data/train2_'+LABEL+'_labels.csv',
                    trans=data_transform['test'], train=True)
# train2 = ImgDataset(label_file='./data/train2_'+LABEL+'_labels.csv',
#                     trans=data_transform['test'], train=False)

# train1_dataloader = DataLoader(train1, batch_size=BATCH_SIZE, shuffle=True)
train1_dataloader = DataLoader(train1, batch_size=BATCH_SIZE, shuffle=False)
train2_dataloader = DataLoader(train2, batch_size=BATCH_SIZE, shuffle=True)
# train2_dataloader = DataLoader(train2, batch_size=BATCH_SIZE, shuffle=False)
