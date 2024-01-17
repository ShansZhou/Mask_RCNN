import torch.nn as nn


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
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes, eps=0.001, momentum=0.01)
        
        # 1x1, Chs*4
        self.conv3 = nn.Conv2d(planes, planes*ResBlock.expansion, kernel_size=3, stride=stride)
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
        out = self.padding2(out)
        
        out = self.conv2(out)
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

class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        
        self.inplanes = 64
        self.layers = [3, 4, 6, 3]
        self.block = ResBlock

        # construct C1 ~ C5
        self.C1 = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(self.inplanes, eps=0.001, momentum=0.01),
            nn.ReLU(inplace=True),
            SamePad2d(kernel_size=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        
        self.C2 = self.make_layers(self.block, 64, self.layers[0])
        self.C3 = self.make_layers(self.block, 128, self.layers[1], stride=2)
        self.C4 = self.make_layers(self.block, 256, self.layers[2], stride=2)
        self.C5 = self.make_layers(self.block, 512, self.layers[3], stride=2)
    
    
    def make_layers(self, block, planes, blocks, stride=1):
        downsample = None
        out_planes = planes*ResBlock.expansion
        if stride != 1 or self.inplanes != out_planes:
            downsample == nn.Sequential(
                nn.Conv2d(self.inplanes, out_planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.01)
            )
        
        layers = []
        # build first block whose stride is different from other blocks
        # stride = 2: downsample the featmap
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * ResBlock.expansion # self.inplanes changed when calling make_layers, TODO
        
        # build rest blocks: layers - 1, stride==1
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        
        # mutiple layers input
        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        
        # [1,3,416,416]->[1,64,104,104]
        out = self.C1(x)
        
        # [1,64,104,104]->[1,256,52,52]
        out = self.C2(out)
       
        # [1,256,52,52]->[1,512,26,26]
        out = self.C3(out)
        
        # [1,512,26,26]->[1,1024,13,13]
        out = self.C4(out)
        
        # [1,1024,13,13]->[1,2048,6,6]
        out = self.C5(out)
        
        return out
    
    def stages(self):
        return [self.C1, self.C2, self.C3, self.C4, self.C5]
        
        