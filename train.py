import numpy as np
import time

import utils.voc_dataset as dt_loader
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from nets.MaskRCNN import MaskRcnn

torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":
    
    Cuda = True
    
    ########### Loading data

    # read classes
    batch_size      = 1
    class_names, num_classes = dt_loader.VOC_CLASSES, len(dt_loader.VOC_CLASSES)

    # model data
    input_shape     = [416, 416]
    model           = MaskRcnn(batch_size, input_shape)
    model_train     = model.train()
    if Cuda:
        model_train     = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train     = model_train.cuda()
        
    # training settings
    epoch_total     = 1
    learn_rate      = 1e-3
    optimizer       = optim.Adam(model_train.parameters(), learn_rate, weight_decay = 5e-4)
    
    # load dataset
    dataset_path = "data/VOCdevkit/VOC2012"
    train_dataset = dt_loader.loadData_voc2012(     data_dir=dataset_path,
                                                    input_shape=input_shape,
                                                    batch_size=batch_size,
                                                    train=True)

    loss = 0.0
    for epoch in range(epoch_total):
        print("----------------------epoch[%d]----------------------" % (epoch))
        for iteration, batch in enumerate(train_dataset):
            
            images, GTs = batch[0], batch[1]
            
            with torch.no_grad():
                if Cuda:
                    images_t  = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                else:
                    images_t  = torch.from_numpy(images).type(torch.FloatTensor)
                    
            # clear gradients
            optimizer.zero_grad()
            
            # forwarding
            # [B,H,W,C] -> [B,C,W,H]
            images_t = images_t.transpose(1,3)
            # [B,C,W,H] -> [B,C,H,W]
            images_t = images_t.transpose(2,3)
            
            # outputs = [rpn_logits, rpn_bbox, classifNet_logits, classifNet_bboxes, mask]
            outputs = model_train(images_t)
            
            # calculate loss
            loss = model.loss(outputs, GTs)
            loss = loss/batch_size
            # BP
            loss.backward()
            print("loss: %.3f" % loss)
            optimizer.step()
    
        print("training is finished")
        torch.save(model.state_dict(),'model_data/trained_models/ep%03d-loss%.3f.pth' % (epoch + 1, loss))
    
        
            