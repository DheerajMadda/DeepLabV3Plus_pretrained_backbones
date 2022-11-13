import torch
import torch.nn as nn
from torchvision.models import resnet50

def get_pretrained_resnet50(include_top=True, freeze_backbone=False):
    resnet50_model = resnet50(weights="IMAGENET1K_V2")
    if freeze_backbone:
        for param in resnet50_model.parameters():
            param.requires_grad = False
    if not include_top:
        resnet50_model = nn.Sequential(*list(resnet50_model.children())[:-2])
    resnet50_model.train()
    return resnet50_model
    


class convolution_block(nn.Module):
    """ 
    Convolutional block
    """
    def __init__(
        self, in_c, out_c, kernel_size=3, padding=1, stride=1, dilation=1, bias=True, dropout=False, dropout_rate=0.0):
        super().__init__()
        
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding, 
            stride=stride, 
            dilation=dilation, 
            bias=bias
        )
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
        self.is_dropout = dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        if self.is_dropout:
            x = self.dropout(x)

        return x

    
class ASPP(nn.Module):
    """ 
    Atrous Spatial Pyramid Pooling:
    """
    def __init__(self, in_c, out_c): 
        super().__init__()
        
        self.conv = convolution_block(in_c, out_c, kernel_size=1, padding=0, bias=False, dropout=False, dropout_rate=0.1)
        
        self.x_1 = convolution_block(out_c, out_c, kernel_size=1, padding=0, dilation=1, bias=False, dropout=True, dropout_rate=0.1)
        self.x_6 = convolution_block(out_c, out_c, kernel_size=3, padding=6, dilation=6, bias=False, dropout=True, dropout_rate=0.15)
        self.x_12 = convolution_block(out_c, out_c, kernel_size=3, padding=12, dilation=12, bias=False, dropout=True, dropout_rate=0.2)
        self.x_18 = convolution_block(out_c, out_c, kernel_size=3, padding=18, dilation=18, bias=False, dropout=True, dropout_rate=0.25)
        
    def forward(self, inputs):
        dim = inputs.shape
        
        x = nn.AvgPool2d(kernel_size=(dim[-2], dim[-1]))(inputs)
        x = self.conv(x)
        x_up = nn.UpsamplingBilinear2d(size=(dim[-2]//x.shape[-2], dim[-1]//x.shape[-1]))(x)
        
        x_1 = self.x_1(inputs)
        x_6 = self.x_6(inputs)
        x_12 = self.x_12(inputs)
        x_18 = self.x_18(inputs)   
        
        return x_up, x_1, x_6, x_12, x_18
    
    
class encoder(nn.Module):
    def __init__(self, in_c, out_c, backbone): 
        super().__init__()
        
        self.feature_extractor_a = nn.Sequential(*list(backbone.children())[:5])
        self.feature_extractor_b = nn.Sequential(*list(backbone.children())[:7])
        self.aspp = ASPP(in_c, out_c)
        self.conv = convolution_block(out_c*5, out_c, kernel_size=1, padding=0, bias=False, dropout=True, dropout_rate=0.1)
        
    def forward(self, inputs):
        
        output_a = self.feature_extractor_a(inputs)
        x = self.feature_extractor_b(inputs)
        x_up, x_1, x_6, x_12, x_18 = self.aspp(x)
        x_cat = torch.cat([x_up, x_1, x_6, x_12, x_18], dim=1)
        output_b = self.conv(x_cat)
        
        return output_a, output_b
    

class decoder(nn.Module):
    def __init__(self, dim_a, num_classes):
        super().__init__()
        
        self.conv_a = convolution_block(dim_a[0], dim_a[1], kernel_size=1, padding=0, bias=False, dropout=True, dropout_rate=0.15)
        self.conv1 =  convolution_block(dim_a[0]*4 + dim_a[1], dim_a[0], kernel_size=3, padding=1, bias=False, dropout=True, dropout_rate=0.1)
        self.conv2 =  convolution_block(dim_a[0], dim_a[0], kernel_size=3, padding=1, bias=False, dropout=True, dropout_rate=0.05)
        self.conv3 = convolution_block(dim_a[0], num_classes, kernel_size=1, padding=0, bias=False, dropout=False, dropout_rate=0.0)
        self.up = nn.UpsamplingBilinear2d(scale_factor=4)
        
    def forward(self, inputs_a, inputs_b):
        x_a = self.conv_a(inputs_a)
        x_b = self.up(inputs_b)
        x_cat = torch.cat([x_a, x_b], dim=1)
        
        x = self.conv1(x_cat)
        x = self.conv2(x)
        x = self.up(x)
        x = self.conv3(x)
        
        return x
        
        
class deeplabv3plus(nn.Module):
    def __init__(self, num_classes=3, freeze_backbone=False):
        super().__init__()
        in_c = out_c = 1024
        backbone = get_pretrained_resnet50(include_top=False, freeze_backbone=freeze_backbone)
        self.encoder = encoder(in_c, out_c, backbone)
        self.decoder = decoder((in_c//4, out_c//16), num_classes)
        
    def forward(self, inputs):
        output_a, output_b = self.encoder(inputs)
        out = self.decoder(output_a, output_b)
        return out
        
        
       
model = deeplabv3plus(num_classes=3)
