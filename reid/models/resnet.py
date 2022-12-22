from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.detach(), a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.detach(), a=0, mode='fan_out')
        init.constant_(m.bias.detach(), 0.0)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.detach(), 1.0, 0.02)
        init.constant_(m.bias.detach(), 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=8):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False), 
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()  
    
    def forward(self, x) :
        max_result=self.maxpool(x) 
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out) 
        return output  

class Fuse(nn.Module):
    def __init__(self):
        super(Fuse, self).__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.bn_1 = nn.BatchNorm1d(2048)
        self.bn_2 = nn.BatchNorm1d(2048)
        self.bn_3 = nn.BatchNorm1d(2048)

        # initialize
        self.bn_1.bias.requires_grad_(False)
        init.constant_(self.bn_1.weight, 1)
        init.constant_(self.bn_1.bias, 0)
        self.bn_2.bias.requires_grad_(False)
        init.constant_(self.bn_2.weight, 1)
        init.constant_(self.bn_2.bias, 0)
        self.bn_3.bias.requires_grad_(False)
        init.constant_(self.bn_3.weight, 1)
        init.constant_(self.bn_3.bias, 0)

    def forward(self, x, ca_upper, ca_low) :
  
        x_embed_upper = x*ca_upper  
        x_embed_low = x*ca_low

        x_embed_x_1 = self.maxpool(x) + self.avgpool(x)
        x_embed_x_1 = x_embed_x_1.view(x_embed_x_1.size(0), -1)
        x_embed_x_1 = self.bn_1(x_embed_x_1)

        x_embed_upper_pool = self.maxpool(x_embed_upper) + self.avgpool(x_embed_upper)
        x_embed_upper_pool = x_embed_upper_pool.view(x_embed_upper_pool.size(0), -1)
        x_embed_upper_pool = self.bn_2(x_embed_upper_pool)

        x_embed_low_pool = self.maxpool(x_embed_low) + self.avgpool(x_embed_low)
        x_embed_low_pool = x_embed_low_pool.view(x_embed_low_pool.size(0), -1)
        x_embed_low_pool = self.bn_3(x_embed_low_pool)

        return [x_embed_x_1, x_embed_upper_pool, x_embed_low_pool] 

class ResNet(nn.Module):
    __factory = {
        18: torchvision.models.resnet18,
        34: torchvision.models.resnet34,
        50: torchvision.models.resnet50,
        101: torchvision.models.resnet101,
        152: torchvision.models.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=700, num_split=1, extract_feat=False):
        super(ResNet, self).__init__()
        self.pretrained = pretrained
        self.depth = depth
        self.cut_at_pooling = cut_at_pooling
        self.num_split = num_split
        self.extract_feat = extract_feat

        # Construct base (pretrained) resnet
        if depth not in ResNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        resnet = ResNet.__factory[depth](pretrained=pretrained)
        resnet.layer4[0].conv2.stride = (1,1)
        resnet.layer4[0].downsample[0].stride = (1,1)

        self.base = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, 
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.ca_upper = ChannelAttention(2048, reduction=4)  
        self.ca_low = ChannelAttention(2048, reduction=4)  

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = resnet.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
                self.feat_bn = nn.BatchNorm1d(self.num_features)

                self.BN_g = nn.BatchNorm1d(self.num_features)
                self.BN_u = nn.BatchNorm1d(self.num_features)
                self.BN_l = nn.BatchNorm1d(self.num_features)

            self.BN_g.bias.requires_grad_(False)
            self.BN_u.bias.requires_grad_(False)
            self.BN_l.bias.requires_grad_(False)

            self.feat_bn.bias.requires_grad_(False)
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)   
                self.adativeFC_upper = nn.Linear(self.num_features*2, self.num_features)    ## update for better results   
                self.adativeFC_low = nn.Linear(self.num_features*2, self.num_features)

                init.normal_(self.classifier.weight, std=0.001) 
                init.kaiming_normal_(self.adativeFC_upper.weight, mode='fan_out')
                init.constant_(self.adativeFC_upper.bias, 0)
                init.kaiming_normal_(self.adativeFC_low.weight, mode='fan_out')
                init.constant_(self.adativeFC_low.bias, 0)

        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)
        init.constant_(self.BN_g.weight, 1)
        init.constant_(self.BN_g.bias, 0)
        init.constant_(self.BN_l.weight, 1)
        init.constant_(self.BN_l.bias, 0)
        init.constant_(self.BN_u.weight, 1)
        init.constant_(self.BN_u.bias, 0)

        if not pretrained:
            self.reset_params()
        

    def forward(self, x):

        x = self.base(x)  # [bs, 2048, 16, 8]

        if self.num_split > 1:
            h = x.size(2)
            x1 = []

            # global feature map
            x_gap = F.avg_pool2d(x, x.size()[2:])  
            x_map = F.max_pool2d(x, x.size()[2:])
            x_embed_x = x_gap + x_map                         
            x_embed_x = x_embed_x.view(x_embed_x.size(0), -1) 
            x_embed_x = self.BN_g(x_embed_x)
            x1.append(x_embed_x)
            # split
            x1_split = [x[:, :, h // self.num_split * s: h // self.num_split * (s+1), :] for s in range(self.num_split)]
            channel_embed_upper = self.ca_upper(x1_split[0]) 
            channel_embed_low = self.ca_low(x1_split[1])

            # upper feature map
            upper = F.avg_pool2d(x1_split[0], x1_split[0].size()[2:])  
            upper_map = F.max_pool2d(x1_split[0], x1_split[0].size()[2:])
            upper_embed_upper_1 = upper + upper_map           
            upper_embed_upper = torch.cat((upper, upper_map), dim=1)  ## update: we change for concat
                      
            # residual structure
            channel_embed_upper_1 = upper_embed_upper_1 * channel_embed_upper  
            # upper_embed_upper = upper_embed_upper_1.view(upper_embed_upper_1.size(0), -1)  
            upper_embed_upper = upper_embed_upper.view(upper_embed_upper.size(0), -1)  ## update: 4096
            upper_embed_upper = self.adativeFC_upper(upper_embed_upper) ##  [bs, 4096]-->[bs, 2048]
            upper_embed_upper = self.BN_u(upper_embed_upper)

            # low feature map
            low = F.avg_pool2d(x1_split[1], x1_split[1].size()[2:])  # [bs, 2048, 1, 1]
            low_map = F.max_pool2d(x1_split[1], x1_split[1].size()[2:])
            low_embed_low_1 = low + low_map
            low_embed_low = torch.cat((low, low_map), dim=1)

            # residual structure
            channel_embed_low_1 = low_embed_low_1 * channel_embed_low
            # low_embed_low = low_embed_low_1.view(low_embed_low_1.size(0), -1)
            low_embed_low = low_embed_low.view(low_embed_low.size(0), -1)
            low_embed_low = self.adativeFC_low(low_embed_low)
            low_embed_low = self.BN_l(low_embed_low)

            x1.append(upper_embed_upper)
            x1.append(low_embed_low)
        else:
            x1 = F.avg_pool2d(x, x.size()[2:])
            x1 = x1.view(x1.size(0), -1)

        if self.extract_feat:
            return x1

        if self.cut_at_pooling:
            return x

        if self.has_embedding:
            x = F.avg_pool2d(x, x.size()[2:])
            x = x.view(x.size(0), -1)
            bn_x = self.feat_bn(self.feat(x))
            bn_x = self.relu(bn_x)
        else:
            x_2 = self.gap(x)
            x_2 = x_2.view(x_2.size(0), -1)
            bn_x = self.feat_bn(x_2)   

        if self.norm:
            bn_x = F.normalize(bn_x)

        if self.dropout > 0:
            bn_x = self.drop(bn_x)

        if self.num_classes > 0:
            prob = self.classifier(x_embed_x)
        else:
            return x, bn_x

        return x1, prob, channel_embed_upper_1, channel_embed_low_1, x 

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class Encoder(nn.Module):
    def __init__(self, model, model_ema):
        super(Encoder, self).__init__()
        self.model = model
        self.model_ema = model_ema
        self.fuse_net = Fuse().cuda()

    def forward(self,input):
        x1, prob, channel_embed_upper_1, channel_embed_low_1,  x = self.model(input) 
        x1_ema, prob_ema,  channel_embed_upper_1_ema, channel_embed_low_1_ema, x_ema = self.model_ema(input)
        outputs = self.fuse_net(x_ema, channel_embed_upper_1, channel_embed_low_1)

        if self.training is False:
            x2 = torch.cat(x1_ema, dim=1)
            x2 = F.normalize(x2)
            return x2, outputs 

        return x1, prob, prob_ema, outputs



def resnet18(**kwargs):
    return ResNet(18, **kwargs)


def resnet34(**kwargs):
    return ResNet(34, **kwargs)


def resnet50(**kwargs):
    return ResNet(50, **kwargs)

def resnet101(**kwargs):
    return ResNet(101, **kwargs)


def resnet152(**kwargs):
    return ResNet(152, **kwargs)
