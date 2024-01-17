import torch.nn as nn
import torch.nn.functional as F

from utils.nn_operators import SamePad2d 

class ResBlock(nn.Module):
    expansion = 4
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        # 1x1, Chs
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes, eps=0.001, momentum=0.01)
        
        # 3x3, Chs
        self.padding2 = SamePad2d(kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(planes, eps=0.001, momentum=0.01)
        
        # 1x1, Chs*4
        self.conv3 = nn.Conv2d(planes, planes*ResBlock.expansion, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(planes*ResBlock.expansion, eps=0.001, momentum=0.01)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample # depends:bottleneck down or bottleneck norm, stride is 2 or 1 in residual
        self.stride = stride
        
    def forward(self, x):
        
        residual = x
        
        # [H,W,chs] ->[H,W,chs]
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # [H,W,chs] ->[H,W,chs]
        out = self.conv2(self.padding2(out))
        out = self.bn2(out)
        out = self.relu(out)
        
        # [H,W,chs] ->[H,W,chs*4]
        out = self.conv3(out)
        out = self.bn3(out)
        
        # [H,W,chs] -> [H,W,chs*4]
        if self.downsample is not None:
            residual = self.downsample(x)
            
        # [H,W,chs] + [H,W,chs*4]
        out = out + residual
        out = self.relu(out)
        
        return out

class Resnet50(nn.Module):
    def __init__(self, out_chs):
        super(Resnet50, self).__init__()
        
        self.inplanes = 64
        self.layers = [3, 4, 6, 3]
        self.block = ResBlock
        self.out_chs = out_chs
        
        ############## bottom-up
        # construct C1 ~ C5
        self.C1 = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(self.inplanes, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            SamePad2d(kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.C2 = self.make_layers(self.block,  64, self.layers[0])
        self.C3 = self.make_layers(self.block, 128, self.layers[1], stride=2)
        self.C4 = self.make_layers(self.block, 256, self.layers[2], stride=2)
        self.C5 = self.make_layers(self.block, 512, self.layers[3], stride=2)
        
        ############## Top-down: FPN
        # 0.5 downsampling
        self.P6 = nn.MaxPool2d(kernel_size=1,stride=2)
        
        # P5
        self.P5_conv1 = nn.Conv2d(2048, self.out_chs, kernel_size=1, stride=1)
        self.P5_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_chs, self.out_chs, kernel_size=3, stride=1)
        )
        
        # P4
        self.P4_conv1 = nn.Conv2d(1024, self.out_chs, kernel_size=1, stride=1)
        self.P4_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_chs, self.out_chs, kernel_size=3, stride=1)
        )
        
        # P3
        self.P3_conv1 = nn.Conv2d(512, self.out_chs, kernel_size=1, stride=1)
        self.P3_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_chs, self.out_chs, kernel_size=3, stride=1)
        )
        
        # P2
        self.P2_conv1 = nn.Conv2d(256, self.out_chs, kernel_size=1, stride=1)
        self.P2_conv2 = nn.Sequential(
            SamePad2d(kernel_size=3, stride=1),
            nn.Conv2d(self.out_chs, self.out_chs, kernel_size=3, stride=1)
        )
    
    
    def make_layers(self, block, planes, blocks, stride=1):
        downsample = None
        out_planes = planes*block.expansion
        if stride != 1 or self.inplanes != out_planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, out_planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.01)
            )
        
        layers = []
        # build first block whose stride is different from other blocks
        # stride = 2: downsample the featmap
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = out_planes # self.inplanes changed when calling make_layers, TODO
        
        # build rest blocks: layers - 1, stride==1
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        
        # mutiple layers input
        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        
        # [1,3,416,416]->[1,64,104,104]
        out = self.C1(x)
        
        # [1,64,104,104]->[1,256,104,104]
        out_c2 = self.C2(out)
       
        # [1,256,104,104]->[1,512,52,52]
        out_c3 = self.C3(out_c2)
        
        # [1,512,52,52]->[1,1024,26,26]
        out_c4 = self.C4(out_c3)
        
        # [1,1024,26,26]->[1,2048,13,13]
        out_c5 = self.C5(out_c4)
        
        # FPN
        # P5 
        m5 = self.P5_conv1(out_c5)
        p5 = self.P5_conv2(m5)
        
        # P4
        m4 = self.P4_conv1(out_c4) + F.upsample(m5, scale_factor=2)
        p4 = self.P4_conv2(m4)
        
        # P3
        m3 = self.P3_conv1(out_c3) + F.upsample(m4, scale_factor=2)
        p3 = self.P3_conv2(m3)
        
        # P2
        m2 = self.P2_conv1(out_c2) + F.upsample(m3, scale_factor=2)
        p2 = self.P2_conv2(m2)
        
        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        p6 = self.P6(p5)
        
        return [p2,p3,p4,p5,p6]
        
        