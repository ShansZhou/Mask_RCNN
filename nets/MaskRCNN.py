

import torch.nn as nn


from nets.ResNet50 import Resnet50


class MaskRcnn(nn.Module):
    def __init__(self):
        super(MaskRcnn, self).__init__()
        
        # init backbone
        self.backbone = Resnet50()
        
        
    def forward(self, x):
        featsmap = self.backbone(x)

        return featsmap