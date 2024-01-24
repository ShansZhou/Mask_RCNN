import numpy as np
import torch.nn as nn
import torch as t
import torch.nn.functional as F
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
    def __init__(self, batch_size, image_shape):
        super(MaskRcnn, self).__init__()
        # batch size
        self.batch_size = batch_size
        # model input shape
        self.image_shape = image_shape
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
        self.prior_anchors = self.genAnchors()
        self.RPN = RPN(num_anchors=self.num_anchors,
                       anchor_stride=self.anchor_stride,
                       chs=self.out_chs)
        
        # init classifier head
        self.cls_poolsize = 7
        self.num_classes = 21
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
            rpn_logits_list = []
            rpn_bbox_list = []
            rpn_rois_list = []
            for i, featsmap in enumerate(featsmaps):
                
                featsmap = featsmap[bid].unsqueeze(0)
                
                # generate RPN classes adn RPN bboxes
                [rpn_logits,rpn_soft_class, rpn_bbox] = self.RPN(featsmap)
                rpn_logits_list.append(rpn_logits)
                rpn_bbox_list.append(rpn_bbox)
                # generate anchors [N, x1,y1,x2,y2]
                gl_anchors = self.prior_anchors[i]
                
                # generate proposals [N, (Bid,x1,y1,x2,y2)]
                rpn_rois = self.generateProposals(featsmap.cpu(),
                                                rpn_soft_class.cpu(), 
                                                rpn_bbox.cpu(), 
                                                gl_anchors.cpu()
                                                )
                rpn_rois = rpn_rois.cuda()
                rpn_rois_list.append(rpn_rois)
                
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

            outputs.append([rpn_logits_list, rpn_bbox_list, classifNet_logits, classifNet_bboxes, mask, rpn_rois_list])
            
            
        return outputs
    
    def generateProposals(self, featsmap, rpn_class, rpn_box, gl_anchors):
        
        batchsize = featsmap.shape[0]
        
        # use the foregroud confidence [N, 1]
        scores = rpn_class[...,1]
        
        # box deltas [N, 4]
        deltas = rpn_box
        
        # apply rpnbox to anchors
        anchors = t.zeros_like(gl_anchors)
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
        bboxes_clampped = t.clone(bboxes)
        bboxes_clampped[...,0]  = t.clamp(bboxes[...,0], min=0, max=featsmap.shape[3])
        bboxes_clampped[...,1]  = t.clamp(bboxes[...,1], min=0, max=featsmap.shape[2])

        
        # NMS bboxes
        # decending bboxes and classes, only choose train_pre_nms bboxes 
        descend_ids = scores.argsort(descending=True)[:,:self.train_pre_nms]
        descend_bboxes = t.zeros_like(descend_ids).unsqueeze(2).repeat(1,1,4).float()
        descend_scores = t.zeros_like(descend_ids).float()
        for b in range(batchsize):
            count = 0
            for id in descend_ids[b]:
                descend_bboxes[b][count] = bboxes_clampped[b][id]
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
        
    def genAnchors(self):
        
        prior_anchors = []
        for feat_stride in self.featmap_strides:
            
            ft_H = int(np.ceil(self.image_shape[0]/feat_stride))
            ft_W = int(np.ceil(self.image_shape[1]/feat_stride))
            
            # generate base anchors (x1,y1,x2,y2)
            fm_stride = feat_stride
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
            
            # [anchors, (x1,y1,x2,y2)]
            anchors = t.stack([x1,y1,x2,y2], dim=-1)
            anchors = anchors.view((-1,4))
            
            prior_anchors.append(anchors)
            
        return prior_anchors
    
    def delta_boxes(self, box, box_gt):
        
        width = box[:,2] - box[:,0]
        height = box[:,3] - box[:,1]
        cx = box[:,0] + width*0.5
        cy = box[:,1] + height*0.5
        
        gt_w = box_gt[:,2] - box_gt[:,0]
        gt_h = box_gt[:,3] - box_gt[:,1]
        gt_cx = box_gt[:,0] + gt_w*0.5
        gt_cy = box_gt[:,1] + gt_h*0.5
        
        dx = (gt_cx - cx) / width
        dy = (gt_cy - cy) / height
        dw = t.log(gt_w / width)
        dh = t.log(gt_h / height)
        
        delta = t.stack([dx,dy,dw,dh], dim=1)
        
        return delta
    
    def decodeBoxWithInfo(self, rpn_rois_list, gt_boxes, gt_classes):
        
        
        
        return boxes_delta
    
    # define RPN ground truth from input GTs
    # compare priors anchors with GTs, match front and back, bbox
    def buildRPNtargets(self, gt_boxes):
        rpn_train_anchors_max = 256
        rpn_positive_thres = 0.3
        rpn_noobj_thres = 0.3

        rpn_match_list = []
        rpn_anchors_list = []
        for idx, fm_stride in enumerate(self.featmap_strides):
            p_anchors = self.prior_anchors[idx]
            ## RPN match: 1 positive anchor, -1 negative anchor, 0 neutral
            rpn_match = t.zeros([p_anchors.shape[0]], dtype=t.int32)
            # define prior anchors with obj using IOU
            # gt_boxes:[x1,y1,x2,y2]  anchors:[x1,y1,x2,y2]
            ious_pairs = tvops.box_iou(p_anchors,gt_boxes)
            ious_max_pair, ious_max_ids = t.max(ious_pairs, dim=1)
            rpn_match[ious_max_pair < rpn_noobj_thres] = -1
            rpn_match[ious_max_pair >= rpn_positive_thres] = 1
            
            ## compute box for positive anchors
            # [max anchors per image, (dy,dx,log(dh),log(dw))]
            rpn_anchor = t.zeros((rpn_train_anchors_max,4))
            
            ids = t.where(rpn_match==1)[0]
            count = 0
            for a_i in (ids):
                gt = gt_boxes[ious_max_ids[a_i]]
                prior_anchor = p_anchors[a_i]
                
                # gt: x,y,w,h
                gt_h = gt[2] - gt[0]
                gt_w = gt[3] - gt[1]
                gt_cx = gt[0] + gt_w*0.5
                gt_cy = gt[1] + gt_h*0.5
                
                # anchor: x1,y1,x2,y2 -> x,y,w,h
                a_h = prior_anchor[2] - prior_anchor[0]
                a_w = prior_anchor[3] - prior_anchor[1]
                a_cx = prior_anchor[0] + a_w*0.5
                a_cy = prior_anchor[1] + a_h*0.5
                
                a_h = t.clip(a_h, 0.0, a_h)
                a_w = t.clip(a_w, 0.0, a_w)
                
                rpn_anchor[count] = t.tensor([
                    (gt_cx - a_cx) / a_w,
                    (gt_cy - a_cy) / a_h,
                    t.log(gt_w/a_w),
                    t.log(gt_h/a_h)
                ])
                
                count+=1
                if count>= rpn_train_anchors_max:break
                
            rpn_match_list.append(rpn_match)
            rpn_anchors_list.append(rpn_anchor)
            
            
        return rpn_match_list, rpn_anchors_list
    
    def rpn_bboxes_loss(self,rpn_match_list, target_rpn_bbox_list, rpn_bbox_list):
        
        loss = 0.0
        for idx, fm_stride in enumerate(self.featmap_strides):
            
            rpn_match = rpn_match_list[idx].cuda()
            pos_ids = t.nonzero(rpn_match==1)
            if len(pos_ids.data[:,0]) ==0:continue
            
            target_box = target_rpn_bbox_list[idx].cuda()
            pred_box = rpn_bbox_list[idx].squeeze(0)
            
            pred_box = pred_box[pos_ids.data[0]]
            target_box = target_box[:pred_box.size()[0],:]
            loss = loss + F.smooth_l1_loss(pred_box, target_box)
            
        
        return loss
    
    def rpn_classes_loss(self, rpn_match_list, rpn_logits_list):
        loss = 0.0
        for idx, fm_stride in enumerate(self.featmap_strides):
            rpn_match = rpn_match_list[idx].cuda()
            rpn_logits = rpn_logits_list[idx].squeeze(0)
                
            # define same shape to pred classess, filling value wrt rpn_match
            target_classes = t.zeros_like(rpn_match,dtype=t.long)
            pos_ids = t.nonzero(rpn_match==1)
            
            if pos_ids.size(0) == 0:                    
                target_classes[:] = 0                
            elif pos_ids.size(0) == len(rpn_match):    
                target_classes[:] = 1               
            else:
                target_classes[rpn_match==1] = 1

            loss = loss + F.cross_entropy(rpn_logits,target_classes)
            
        return loss
    
    # find best matched boxes from preds of classifNet
    def matchPredbox_targets(self, gt_boxes, rpn_rois_list, gt_classes, gt_mask):
        
        roi_positive_thres = 0.3
        gt_boxes = gt_boxes.cuda()
        # for current rpn_rois, seperate rois into positive or negative wrt max IOU
        # when a rpn_roi is highly overlaping with one of gt_boxes, it is a positive roi.
        # otherwise, it is negative
        rpn_rois = t.cat(rpn_rois_list, dim=0)[:,1:].cuda()
        ious = tvops.box_iou(rpn_rois, gt_boxes)
        # iou_max: [N, max_iou], ioumax_ids: [N, max_id_gtbox] 
        iou_max, ioumax_ids = t.max(ious, dim=1)
        
        positive_roi_mask = iou_max >= roi_positive_thres
        negative_roi_mask = iou_max < roi_positive_thres
        pos_num = t.nonzero(positive_roi_mask).size(0)
        neg_num = t.nonzero(negative_roi_mask).size(0)
        
        pos_box_delta = t.zeros((pos_num,4)).cuda()
        # negative obj is not counting box loss
        neg_box_delta = t.zeros((neg_num,4)).cuda()
        target_classes = t.zeros((pos_num+neg_num), dtype=t.long).cuda()
        
        if pos_num:
            # for each pos roi, find its related best gtbox
            pos_rois = rpn_rois[positive_roi_mask]
            best_gtbox_ids = ioumax_ids[positive_roi_mask]
            best_gtboxes = gt_boxes[best_gtbox_ids]
            best_gtboxes_classes = gt_classes[best_gtbox_ids].long()
            # pos_rois and best_gtboxes are same shape: one to one mapping
            pos_box_delta = self.delta_boxes(pos_rois, best_gtboxes)
            
            # targets classes
            target_classes[:pos_num] = best_gtboxes_classes
        
        boxes_delta = t.cat([pos_box_delta, neg_box_delta], dim=0)
        
        return boxes_delta, target_classes, target_masks

    # crop roi from mask and resize to maskNet output scale
    def get_target_masks(self,rois, gt_classes, gt_mask):
        pass
    
    def clsNet_classes_loss(self, classifNet_logits, clsNet_targets_classes):
        pred = t.clip(classifNet_logits[:len(clsNet_targets_classes)], 1e-7, 1.0-1e-7)
        loss = F.cross_entropy(pred, clsNet_targets_classes)
        return loss
    
    def clsNet_bboxes_loss(self, classifNet_bboxes, clsNet_boxes_delta):
        preds_boxes_delta = classifNet_bboxes[:len(clsNet_boxes_delta),0,:].squeeze(1)
        loss = F.smooth_l1_loss(preds_boxes_delta, clsNet_boxes_delta)
        return loss
    
    def maskNet_loss(self, pred_mask, gt_mask):
        loss=0.0
        
    
    def loss(self, outputs, GTs):
        
        loss_total = 0.0
        for bid in range(self.batch_size):
            
            # decoding the prediction from forwarding to source
            rpn_logits_list, rpn_bbox_list, classifNet_logits, classifNet_bboxes, pred_mask, rpn_rois_list = outputs[bid]
            
            # detections = self.decodeBoxWithInfo()
            
            # decode the ground truth
            targets_dict = GTs[bid]
            
            gt_boxes = t.tensor(targets_dict["boxes"])
            
            gt_classes = t.tensor(targets_dict["labels"])
            
            gt_mask = targets_dict["masks"]
            
            # generate rpn ground truth targets
            rpn_match_list, target_rpn_bbox_list = self.buildRPNtargets(gt_boxes)
            # RPN logits loss
            loss_rpn_classes = self.rpn_classes_loss(rpn_match_list,rpn_logits_list)
            # RPN bbox loss
            loss_rpn_box = self.rpn_bboxes_loss(rpn_match_list, target_rpn_bbox_list,rpn_bbox_list)
            
            # find best matched pred boxes to gt_boxes
            clsNet_boxes_delta, clsNet_targets_classes, target_masks = self.matchPredbox_targets(gt_boxes, rpn_rois_list, gt_classes, gt_mask)
             # ClassifNet bbox loss
            loss_clsNet_bboxes = self.clsNet_bboxes_loss(classifNet_bboxes, clsNet_boxes_delta)
            # ClassifNet logits loss
            loss_clsNet_classes = self.clsNet_classes_loss(classifNet_logits, clsNet_targets_classes)
           
            # MaskNet loss 
            loss_maskNet_mask = self.maskNet_loss(pred_mask, target_masks)
            
            loss_total = loss_total + loss_rpn_box+loss_rpn_classes+loss_clsNet_classes+loss_clsNet_bboxes
            
        
        return loss_total