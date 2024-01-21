import os
import cv2 as cv
import xml.etree.ElementTree as ET
import numpy as np

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader




VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
)

class VOCDataset(Dataset):
    # download VOC 2012: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
    def __init__(self, data_dir, input_shape, split, train):
            super(VOCDataset, self).__init__()
            
            self.data_dir = data_dir
            self.input_shape = input_shape
            self.split = split
            self.train = train
            
            # instances segmentation task
            id_file = os.path.join(data_dir, "ImageSets/Segmentation/{}.txt".format(split))
            self.ids = [id_.strip() for id_ in open(id_file)]
            
            # classes's values must start from 1, because 0 means background in the model
            self.classes = {i: n for i, n in enumerate(VOC_CLASSES, 1)}
            
            
    
    def __getitem__(self, i):
        img_id = self.ids[i]
        image, target = self.get_target(img_id)
        
        # check if the preprocess is correct
        # check_data(image, target)
        
        return image, target 
        
    def __len__(self):
        return len(self.ids)          

    def get_target(self, img_id):
        img_path = os.path.join(self.data_dir, "JPEGImages/{}.jpg".format(img_id))
        image = cv.imread(img_path, cv.IMREAD_COLOR)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        mask_path = os.path.join(self.data_dir, 'SegmentationObject/{}.png'.format(img_id))
        masks = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)

        anno = ET.parse(os.path.join(self.data_dir, "Annotations", "{}.xml".format(img_id)))
        boxes = []
        labels = []
        for obj in anno.findall("object"):
            bndbox = obj.find("bndbox")
            bbox = [int(bndbox.find(tag).text) for tag in ["xmin", "ymin", "xmax", "ymax"]]
            name = obj.find("name").text
            label = VOC_CLASSES.index(name) + 1

            boxes.append(np.expand_dims(bbox,0))
            labels.append(label)

        boxes = np.concatenate(boxes, axis=0)
        
        labels = np.array(labels)

        img_id = np.array([self.ids.index(img_id)])
        
        # preprocess
        image, boxes, masks = self.preprocess(image, boxes, masks)
        
        target = dict(image_id=img_id, boxes=boxes, labels=labels, masks=masks)
        return image, target

    # every batch has same size of images
    def preprocess(self, image, box, masks):
        
        im_H, im_W, chs  = image.shape
        in_H, in_W = self.input_shape
        
        # scale image with fixed ratio
        scalar = min(in_H/im_H, in_W/im_W)
        new_h = int(im_H*scalar)
        new_w = int(im_W*scalar)
        dx = (in_W - new_w)//2
        dy = (in_H - new_h)//2
        
        im_scaled = cv.resize(image, dsize=(new_w, new_h))
        im_input = np.ones((in_H, in_W, 3))*128
        im_input[dy:(dy+new_h), dx:(dx+new_w)] = np.array(im_scaled)
        im_input = np.float32(im_input)
        
        mask_scaled = cv.resize(masks, dsize=(new_w, new_h))
        mask_input = np.zeros((in_H, in_W))
        mask_input[dy:(dy+new_h), dx:(dx+new_w)] = np.array(mask_scaled)
        
        # mapping original box coords to new image
        if len(box)>0:
            box[:, [0,2]] = box[:, [0,2]]*scalar + dx
            box[:, [1,3]] = box[:, [1,3]]*scalar + dy
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>in_W] = in_W
            box[:, 3][box[:, 3]>in_H] = in_H
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

        return im_input, box, mask_input
         
def mrcnn_dataset_collate(batch):
    images = []
    targets = []
    
    for img, target in batch:
        images.append(img)
        targets.append(target)
    
    images = np.array(images)
    return images, targets   

def check_data(image, target):
    
    
    im_disp = cv.cvtColor(image, cv.COLOR_RGB2BGR) /255.0
    boxes = target["boxes"]
    mask = target["masks"]
    
    
    for box in boxes:
        cv.rectangle(im_disp,box[:2], box[2:],(0,255,0),2)
    
    cv.imshow("im", im_disp)
    cv.imshow("mask", mask/255.0)
    
    cv.waitKey(0)
    
def loadData_voc2012(data_dir, input_shape, batch_size=1, train=True):
    
    train_or_val = "train" if train else "val"
    
    dataset = VOCDataset(data_dir=data_dir, input_shape=input_shape, split=train_or_val, train=train)
    
    gen           = DataLoader(dataset=dataset, shuffle = True, batch_size = batch_size, num_workers = 0, pin_memory=True,
                                drop_last=True, collate_fn=mrcnn_dataset_collate)
    
    return gen