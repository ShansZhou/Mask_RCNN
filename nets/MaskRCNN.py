

import torch.nn as nn


from nets.ResNet50 import Resnet50
from utils.nn_operators import SamePad2d

class MaskRcnn(nn.Module):
    def __init__(self):
        super(MaskRcnn, self).__init__()
        
        # init backbone
        self.backbone = Resnet50(256)
        
        
    def forward(self, x):
        featsmaps = self.backbone(x)

        return featsmaps