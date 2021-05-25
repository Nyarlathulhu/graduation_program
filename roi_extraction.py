import cv2
import numpy as np
import os


def fillHole(srcBin):
    temp = srcBin.copy()
    cv2.floodFill(temp, None, (0, 0), 255)
    # invert image
    temp = cv2.bitwise_not(temp)
    # cv2.imshow('temp', temp)

    # invert image
    out = srcBin | temp

    return out


def getRoi(im_out, img):
    """
        im_out: preprocessed binary image
        img: original, type BGR
    """
    num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(im_out, connectivity=8, ltype=cv2.CV_32S)
    image = np.copy(img)
    roi_list = []
    for t in range(1, num_labels, 1):
        x, y, w, h, area = stats[t]
        if area < 500:
            continue
        cx, cy = centers[t]
        # find center
        cv2.circle(image, (np.int32(cx), np.int32(cy)), 2, (0, 255, 0), 2, 8, 0)
        # outer rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2, 8, 0)
        # save coordinate and width and height
        roi_list.append((x, y, w, h))

    return num_labels, labels, image, roi_list


def colorImgShow(im_out, num_labels, labels):
    """
        im_out: preprocessed binary image
        num_labels: num of components
        labels: output flag image，with background index = 0
    """
    # make the colors
    colors = []
    for i in range(num_labels):
        b = np.random.randint(0, 256)
        g = np.random.randint(0, 256)
        r = np.random.randint(0, 256)
        colors.append((b, g, r))
    colors[0] = (0, 0, 0)

    # draw the image
    h, w = im_out.shape
    image = np.zeros((h, w, 3), dtype=np.uint8)
    for row in range(h):
        for col in range(w):
            image[row, col] = (colors[labels[row, col]])

    return image


def saveRoi(src, roi_list, name):
    """
        src: copy of original ones
        roi_list: save position info
    """
    for i in range(len(roi_list)):
        x, y, w, h = roi_list[i]
        roi = src[y:y+h, x:x+w]
        cv2.imwrite('./roi_image/train2/sleeve_length/{}'.format(name), roi)
        np.save('./results/roi_pos/train2/sleeve_length/{}.npy'.format(name), roi_list[i])
        # print("%d Finished! " % i)


PATH = './grad_aam_image/train2/sleeve_length/'
images = []
i = 0

for root, dirs, filename in os.walk(PATH):
    print(filename)
for img in filename:
    images.append(cv2.imread(PATH + img, 1))

for img in images:
    # preprocess
    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_th = cv2.threshold(img_in, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # white for front, black for background
    img_th = cv2.bitwise_not(img_th)

    img_out = fillHole(img_th)
    num_labels, labels, image, roi_list = getRoi(img_th, img)
    colored_img = colorImgShow(img_out, num_labels, labels)
    saveRoi(img, roi_list, filename[i])
    i += 1
