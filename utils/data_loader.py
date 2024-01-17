import numpy as np
import cv2 as cv

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

class YoloDataset(Dataset):
    
    def __init__(self, annotation_lines, input_shape, num_classes, train):
        super().__init__()
        
        self.annotation_lines   = annotation_lines
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.length             = len(self.annotation_lines)
        self.train              = train
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        index = index % self.length
        image, box  = self.get_image_data(self.annotation_lines[index], self.input_shape[0:2])
        
        if len(box) !=0:
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]

            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2.0
        
        return image, box
    
    def get_image_data(self, annotation_line, input_shape):
        line    = annotation_line.split()
        
        im_path = "data/VOC2007/JPEGImages/"+line[0].split('/')[-1]
        
        image = cv.imread(im_path, cv.IMREAD_COLOR)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        im_H, im_W, chs = image.shape
        in_H, in_W = input_shape
        
        box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]], np.float32)
        
        # scale image with fixed ratio
        scalar = min(in_H/im_H, in_W/im_W)
        new_h = int(im_H*scalar)
        new_w = int(im_W*scalar)
        dx = (in_W - new_w)//2
        dy = (in_H - new_h)//2
        
        im_scaled = cv.resize(image, dsize=(new_w, new_h))
        im_input = np.ones((in_H, in_W, 3))*128
        
        # copy scaled img to input img
        im_input[dy:(dy+new_h), dx:(dx+new_w)] = np.array(im_scaled)
        im_input = np.float32(im_input)
        
        # mapping original box coords to new image
        if len(box)>0:
            box[:, [0,2]] = box[:, [0,2]]*new_w/in_W + dx
            box[:, [1,3]] = box[:, [1,3]]*new_h/in_H + dy
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>in_W] = in_W
            box[:, 3][box[:, 3]>in_H] = in_H
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

        return im_input, box


def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    return images, bboxes

def conversion_NCWH(batch_data):
    return np.transpose(batch_data, (0,3,1,2))

def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

def loadData(annotation_path, input_shape, num_classes, batch_size):
    
    # read dataset annotation
    train_annotation_path   = annotation_path+"/2007_train.txt"
    val_annotation_path     = annotation_path+"/2007_val.txt"

    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines   = f.readlines()
        
    # train_dataset   = YoloDataset(train_lines, input_shape, num_classes, True)
    train_dataset   = YoloDataset(train_lines[:200], input_shape, num_classes, True)
    val_dataset     = YoloDataset(val_lines, input_shape, num_classes, False)

    gen             = DataLoader(train_dataset, shuffle = True, batch_size = batch_size, num_workers = 1, pin_memory=True,
                            drop_last=True, collate_fn=yolo_dataset_collate)
    gen_val         = DataLoader(val_dataset  , shuffle = True, batch_size = batch_size, num_workers = 1, pin_memory=True, 
                                drop_last=True, collate_fn=yolo_dataset_collate)
    
    
    return gen, gen_val