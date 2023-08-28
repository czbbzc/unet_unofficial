import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import torchvision

def weights_init1(layer):
    if isinstance(layer, nn.Conv2d):
        N = layer.kernel_size[0] * layer.kernel_size[1] * layer.in_channels
        nn.init.normal_(layer.weight,std=N**0.5)

class conv3_3(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU()
        
        self.apply(weights_init1) 

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return self.act2(x)
    
class crop_concat(nn.Module):

    def forward(self, x, x_before):

        x_crop = F.center_crop(x_before, [x.shape[2], x.shape[3]])
        return torch.cat([x, x_crop], dim=1)
    
class up_conv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=1, padding=0)

    def forward(self, x):

        x = self.up(x)
        x = self.conv(x)
        t = torchvision.transforms.Resize(size=(x.shape[2]+1, x.shape[3]+1))        
        x = t(x)
        
        return x
    

class unet(nn.Module):
    
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv33_1  = conv3_3(in_channels=in_channels, out_channels=64)
        self.conv33_2  = conv3_3(in_channels=64, out_channels=128)
        self.conv33_3  = conv3_3(in_channels=128, out_channels=256)
        self.conv33_4  = conv3_3(in_channels=256, out_channels=512)
        self.conv33_5  = conv3_3(in_channels=512, out_channels=1024)
        
        self.max_pool = nn.MaxPool2d(2)
        
        self.crop_concat = crop_concat()
        
        self.up_conv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0)
        self.up_conv11 = up_conv(1024, 512)
        self.conv33_6  = conv3_3(in_channels=1024, out_channels=512)
        
        self.up_conv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)
        self.up_conv22 = up_conv(512, 256)
        self.conv33_7  = conv3_3(in_channels=512, out_channels=256)
        
        self.up_conv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)
        self.up_conv33 = up_conv(256, 128)
        self.conv33_8  = conv3_3(in_channels=256, out_channels=128)
        
        self.up_conv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)
        self.up_conv44 = up_conv(128, 64)
        self.conv33_9  = conv3_3(in_channels=128, out_channels=64)
        
        self.last_conv = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):

        x1 = self.conv33_1(x)
        x = self.max_pool(x1)
        
        x2 = self.conv33_2(x)
        x = self.max_pool(x2)
        
        x3 = self.conv33_3(x)
        x = self.max_pool(x3)
        
        x4 = self.conv33_4(x)
        x = self.max_pool(x4)
        
        x = self.conv33_5(x)
        
        x = self.up_conv11(x)
        x = self.crop_concat(x, x4)
        x = self.conv33_6(x)
        
        x = self.up_conv22(x)
        x = self.crop_concat(x, x3)
        x = self.conv33_7(x)
        
        x = self.up_conv33(x)
        x = self.crop_concat(x, x2)
        x = self.conv33_8(x)
        
        x = self.up_conv44(x)
        x = self.crop_concat(x, x1)
        x = self.conv33_9(x)
        
        x = self.last_conv(x)
        
        return x
    
a = unet(1, 2)
print(a)

# for layer in a.modules():
#     if isinstance(layer, nn.Conv2d):
#         N = layer.kernel_size[0] * layer.kernel_size[1] * layer.in_channels
#         nn.init.normal_(layer.weight,std=N**0.5)
        
        # torch.nn.init.xavier_uniform(layer.weight)
        # layer.bias.data.fill_(2)
