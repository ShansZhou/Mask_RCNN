import numpy as np
import time

import utils.data_loader as dt_loader
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from nets.MaskRCNN import MaskRcnn


if __name__ == "__main__":
    
    Cuda = True
    
    ########### Loading data

    # read classes
    classes_path    = 'model_data/voc_classes.txt'
    class_names, num_classes = dt_loader.get_classes(classes_path)

    # model data
    input_shape     = [416, 416]
    model           = MaskRcnn()
    model_train     = model.train()
    if Cuda:
        model_train     = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train     = model_train.cuda()
        
    # training settings
    epoch_total     = 1
    batch_size      = 2
    learn_rate      = 1e-3
    optimizer       = optim.Adam(model_train.parameters(), learn_rate, weight_decay = 5e-4)
    
    # load dataset
    annotation_path = "data/VOC2007"
    train_dataset, val_dataset = dt_loader.loadData(annotation_path, input_shape, num_classes, batch_size)

    loss = 0.0
    for epoch in range(epoch_total):
        print("----------------------epoch[%d]----------------------" % (epoch))
        for iteration, batch in enumerate(train_dataset):
            
            images, GTs = batch[0], batch[1]
            with torch.no_grad():
                if Cuda:
                    images  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    GTs = [torch.from_numpy(ann).type(torch.FloatTensor).cuda() for ann in GTs]
                else:
                    images  = torch.from_numpy(images).type(torch.FloatTensor)
                    GTs = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in GTs]
                    
            # clear gradients
            optimizer.zero_grad()
            
            # forwarding
            # [B,H,W,C] -> [B,C,W,H]
            images = images.transpose(1,3)
            # [B,C,W,H] -> [B,C,H,W]
            images = images.transpose(2,3)
            
            # outputs = [rpn_logits, rpn_bbox, classifNet_logits, classifNet_bboxes, mask]
            outputs = model_train(images)
            
            # calculate loss
            loss = model.computeLoss(outputs, GTs)
            
            # BP
            loss.backward()
            
            optimizer.step()
    
        print("training is finished")
        torch.save(model.state_dict(),'model_data/trained_models/ep%03d-loss%.3f.pth' % (epoch + 1, loss))
    
        
            