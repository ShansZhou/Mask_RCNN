import torch.nn as nn
import torch as t
import torchvision.ops as tvops

from nets.ResNet50 import Resnet50
from utils.nn_operators import SamePad2d

############################################################
#  Region Proposal Network
############################################################
class RPN(nn.Module):
    def __init__(self, num_anchors=9, anchor_stride=1, chs=256):
        super(RPN, self).__init__()

        self.anchor_stride = anchor_stride
        self.chs = chs
        
        # keep shape same in following convolution
        self.padding = SamePad2d(kernel_size=3, stride=anchor_stride)
        
        # shared conv
        self.conv_shared = nn.Conv2d(self.chs, 512, kernel_size=3, stride=self.anchor_stride)
        self.relu = nn.ReLU(inplace=True)
        
        # class conv
        self.conv_class = nn.Conv2d(512, 2*num_anchors, kernel_size=1, stride=1)
        self.softmax = nn.Softmax(dim=2)
        
        # bbox conv
        self.conv_bbox = nn.Conv2d(512, 4*num_anchors, kernel_size=1, stride=1)
        
    def forward(self, x):
        
        #[B,256,H,W] -> [B,512,H,W]
        x_shared = self.relu(self.conv_shared(self.padding(x)))
        
        ################## Classes: front or back
        # [B, 2*num_anchors, H,W]
        rpn_class = self.conv_class(x_shared)
        
        # reshape [B, anchors, 2]
        rpn_class = rpn_class.permute(0,2,3,1).contiguous().view(x.size()[0],-1,2)
        
        # Softmax on last dimension of BG/FG.
        rpn_soft_class = self.softmax(rpn_class)
        
        
        ################## BBoxes: [x,y,w,h]
        rpn_bbox = self.conv_bbox(x_shared)
        
        # reshape [B, anchors, 4]
        rpn_bbox = rpn_bbox.permute(0,2,3,1).contiguous().view(x.size()[0],-1,4)
        
        
        return [rpn_class, rpn_soft_class, rpn_bbox]


############################################################
#  Classifier Head Network
############################################################
class Classifier(nn.Module):
    def __init__(self, fpn_chs, pool_size, image_shape, num_classes):
        super(Classifier, self).__init__() 

        self.depth = fpn_chs
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.num_classes = num_classes
        
        # layer 1
        self.conv1 = nn.Conv2d(in_channels= fpn_chs, out_channels=1024, kernel_size=pool_size, stride=1)
        self.bn1 = nn.BatchNorm2d(1024,eps=0.001, momentum=0.01)
        
        # layer 2
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(1024,eps=0.001,momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        
        # FC 3
        self.fc = nn.Linear(in_features=1024, out_features=num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.linear_bbox = nn.Linear(in_features=1024, out_features=num_classes*4)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = out.view(-1,1024)
        
        # output 1
        class_logits = self.fc(out)
        scores = self.softmax(class_logits)
        # output 2
        bboxes = self.linear_bbox(out)
        bboxes = bboxes.view(bboxes.size()[0], -1, 4)
        
        return [class_logits, scores, bboxes]
   
        
############################################################
#  Mask Head Network
############################################################
class Mask_Net(nn.Module):
    def __init__(self,fpn_chs, pool_size, image_shape, num_classes):
        super(Mask_Net, self).__init__()
        self.fpn_chs = fpn_chs
        self.pool_size= pool_size
        self.image_shape = image_shape
        self.num_classes = num_classes
        
        self.padding = SamePad2d(kernel_size=3, stride=1)
        
        # Conv layers
        self.CONV_layers = self.make_layers(fpn_chs=fpn_chs,layer_count=4)
        
        # Deconv
        self.deconv = nn.ConvTranspose2d(fpn_chs, fpn_chs, kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        # final conv
        self.final_conv = nn.Conv2d(fpn_chs, num_classes, kernel_size=1, stride=1)
        self.sigmoid = nn.Sigmoid()
        
    def make_layers(self, fpn_chs, layer_count=4):
        
        layers = []
        for l in range(layer_count):
            layer = nn.Sequential(
                SamePad2d(kernel_size=3, stride=1),
                nn.Conv2d(in_channels=fpn_chs, out_channels=fpn_chs, kernel_size=3, stride=1),
                nn.BatchNorm2d(fpn_chs, eps=0.001),
                nn.ReLU(inplace=True)
            )
                
            layers.append(layer)
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        
        out = self.CONV_layers(x)
        
        # deconv
        out = self.deconv(out)
        out = self.relu(out)
        out = self.final_conv(out)
        out = self.sigmoid(out)
        
        return out
        

class MaskRcnn(nn.Module):
    def __init__(self):
        super(MaskRcnn, self).__init__()
        # batch size
        self.batch_size = 2
        # nms_thres
        self.nms_thres = 0.5
        # maximum proposals
        self.train_pre_nms = 100
        self.train_post_nms = 50
        
        # init backbone
        self.scales = []
        self.out_chs = 256
        self.featmap_strides = [4,8,16,32,64]
        self.backbone = Resnet50(self.out_chs)
        
        # init RPN
        self.anchor_scales = t.tensor([4,8,16,32,64]) 
        self.anchor_ratios = t.tensor([0.5,1,2])
        self.num_anchors = len(self.anchor_scales)*len(self.anchor_ratios)
        self.anchor_stride =1
        self.RPN = RPN(num_anchors=self.num_anchors,
                       anchor_stride=self.anchor_stride,
                       chs=self.out_chs)
        
        # init classifier head
        self.cls_poolsize = 7
        self.image_shape = [416, 416]
        self.num_classes = 20
        self.classif_net = Classifier(self.out_chs, self.cls_poolsize, self.image_shape, self.num_classes)
        
        
        # init mask head
        self.mask_net = Mask_Net(self.out_chs,self.cls_poolsize, self.image_shape, self.num_classes)
        
    def forward(self, x):
        
        # featsmap in different scale
        # p2 [B,256,104,104], stride = 4
        # p3 [B,256,52,52],   stride = 8
        # p4 [B,256,26,26],   stride = 16
        # p5 [B,256,13,13],   stride = 32
        # p6 [B,256,7,7],     stride = 64
        featsmaps = self.backbone(x)
        
        outputs = []
        for bid in range(self.batch_size):
            # for each feat map in different scale, compute the proposals
            feats_roialign_pred = []
            feats_roialign_mask = []
            for i, featsmap in enumerate(featsmaps):
                
                featsmap = featsmap[bid].unsqueeze(0)
                
                # generate RPN classes adn RPN bboxes
                [rpn_logits,rpn_soft_class, rpn_bbox] = self.RPN(featsmap)
                
                # generate anchors [B, N, x1,y1,x2,y2]
                gl_anchors = self.genAnchors(i, featsmap)
                
                # generate proposals [N, (Bid,x1,y1,x2,y2)]
                rpn_rois = self.generateProposals(featsmap.cpu(),
                                                rpn_soft_class.cpu(), 
                                                rpn_bbox.cpu(), 
                                                gl_anchors.cpu()
                                                )
                rpn_rois = rpn_rois.cuda()
                
                # Featmap with ROIs
                pred_head_shape = [7,7]
                mask_head_shape = [14,14]
                spatial_scale = 1.0 / self.featmap_strides[i]
                roiAligns_pred = tvops.roi_align(featsmap, rpn_rois,pred_head_shape, spatial_scale)
                roiAligns_mask = tvops.roi_align(featsmap, rpn_rois,mask_head_shape, spatial_scale)
                
                feats_roialign_pred.append(roiAligns_pred)
                feats_roialign_mask.append(roiAligns_mask)
            
            feats_roialign_pred = t.cat(feats_roialign_pred, dim=0)
            feats_roialign_mask = t.cat(feats_roialign_mask, dim=0)
            
            # Classifier and Boxes Regression heads
            [classifNet_logits, classifNet_scores, classifNet_bboxes] = self.classif_net(feats_roialign_pred)
            # Mask detection head
            mask = self.mask_net(feats_roialign_mask)

            outputs.append([rpn_logits, rpn_bbox, classifNet_logits, classifNet_bboxes, mask])
            
            
        return outputs
    
    def generateProposals(self, featsmap, rpn_class, rpn_box, gl_anchors):
        
        batchsize = featsmap.shape[0]
        
        # use the foregroud confidence [B, N, 1]
        scores = rpn_class[...,1]
        
        # box deltas [B, N, 4]
        deltas = rpn_box
        
        # apply rpnbox to anchors
        anchors = gl_anchors
        ## x1,y1,x2,y2 -> x,y,w,h
        ### calc w,h
        anchors[...,2:] = anchors[...,2:] - anchors[...,:2]
        ### calc x,y -> center 
        anchors[...,:2] = anchors[...,:2] + anchors[...,2:] / 2
        
        gx = anchors[...,0] + anchors[...,2]*deltas[...,0]
        gy = anchors[...,1] + anchors[...,3]*deltas[...,1]
        gw = anchors[...,2] * t.exp(deltas[...,2])
        gh = anchors[...,3] * t.exp(deltas[...,3])
        
        bboxes = t.cat([gx.unsqueeze(2),gy.unsqueeze(2),gw.unsqueeze(2),gh.unsqueeze(2)], dim=2)
        
        bboxes[...,:2] = bboxes[...,:2] - bboxes[...,2:]/2
        bboxes[...,2:] = bboxes[...,2:] + bboxes[...,2:]/2
        
        # clamp bboxes inside featmap: 0~W, 0~H
        bboxes[...,0] = t.clamp(bboxes[...,0], min=0, max=featsmap.shape[3])
        bboxes[...,1] = t.clamp(bboxes[...,1], min=0, max=featsmap.shape[2])
        
        
        # NMS bboxes
        # decending bboxes and classes, only choose train_pre_nms bboxes 
        descend_ids = scores.argsort(descending=True)[:,:self.train_pre_nms]
        descend_bboxes = t.zeros_like(descend_ids).unsqueeze(2).repeat(1,1,4).float()
        descend_scores = t.zeros_like(descend_ids).float()
        for b in range(batchsize):
            count = 0
            for id in descend_ids[b]:
                descend_bboxes[b][count] = bboxes[b][id]
                descend_scores[b][count] = scores[b][id]
                count+=1
        
        # for each batch, they have different size of boxes        
        rpn_rois = []
        for b in range(batchsize):
            
            keeps = tvops.batched_nms(boxes=descend_bboxes[b],
                                        scores=descend_scores[b],
                                        idxs=descend_ids[b],
                                        iou_threshold=self.nms_thres)
            rois = descend_bboxes[b][keeps[:self.train_post_nms]]
            roi_idx = t.ones(len(rois), dtype=t.int32)
            roi_idx = t.unsqueeze(roi_idx*b, dim=1)
            rois = t.cat([roi_idx,rois], dim=1)
            
            rpn_rois.append(rois)
            
            
        # [N, Batch_id,x,y,w,h]
        rpn_rois = t.cat(rpn_rois, dim=0)
        
        return rpn_rois
        
    def genAnchors(self, sidx, featsmap):
        b,_,ft_H,ft_W = featsmap.shape
        
        # generate base anchors (x1,y1,x2,y2)
        fm_stride = self.featmap_strides[sidx]
        px = fm_stride/2
        py = fm_stride/2
        base_anchors = t.zeros(self.num_anchors, 4)
        
        for i in range(len(self.anchor_ratios)):
            for j in range(len(self.anchor_scales)):
                
                w = fm_stride*self.anchor_scales[j]*t.sqrt(self.anchor_ratios[i])
                
                h = fm_stride*self.anchor_scales[j]/t.sqrt(self.anchor_ratios[i])
                
                bId = i*len(self.anchor_ratios) + j
                
                base_anchors[bId] = t.Tensor([px - w/2, py - h/2, px + w/2, py + h/2])
                
        
        # generate global anchors assciated with current featmap
        gridY, gridX = t.meshgrid(t.arange(ft_W), t.arange(ft_H))
        gridX, gridY = gridX.contiguous(), gridY.contiguous()
        # the grids on sourceImg with stride, generate anchors at every grid location
        x1 = gridX.view((-1,1))*fm_stride + base_anchors[:,0]
        y1 = gridY.view((-1,1))*fm_stride + base_anchors[:,1]
        x2 = gridX.view((-1,1))*fm_stride + base_anchors[:,2]
        y2 = gridY.view((-1,1))*fm_stride + base_anchors[:,3]
        
        # [anchors, (y1,x1,y2,x2)]
        anchors = t.stack([x1,y1,x2,y2], dim=-1)
        anchors = anchors.view((-1,4))
        
        # [B, anchors, (y1,x1,y2,x2)]
        anchors = anchors.squeeze(0).repeat(b,1,1)
        return anchors
    
    def decodeBoxWithInfo(self, rpn_rois, scores, bboxes, featsmaps):
        return detections
    
    def computeLoss(self, outputs, GTs):
        
        loss_total = 0.0
        for bid in range(self.batch_size):
            
            output = outputs[bid]
            
            targets = GTs[bid]
            
            
            
            # RPN logits loss
            
            # RPN bbox loss
            
            # ClassifNet logits loss
            
            # ClassifNet bbox loss
            
            # MaskNet loss 
            
            
        
        return loss_total / self.batch_size