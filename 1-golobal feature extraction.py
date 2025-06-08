import torch
from torch import nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1 + (dilation - 1), bias=False, dilation=dilation)
        
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)


        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, in_ch, block, layers, strides, dilations):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],
                                       stride=strides[0], dilation=dilations[0])
        self.layer2 = self._make_layer(block, 128, layers[1],
                                       stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(block, 256, layers[2],
                                       stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_layer(block, 512, layers[3],
                                       stride=strides[3], dilation=dilations[3])

    def _make_layer(self, block, planes, blocks, stride=1,
                    dilation=1):

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = planes * block.expansion   # 64 x 4
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        conv1_f = self.relu(x)
        x = self.maxpool(conv1_f)

        layer1_f = self.layer1(x)
        layer2_f = self.layer2(layer1_f)
        layer3_f = self.layer3(layer2_f)
        layer4_f = self.layer4(layer3_f)
        
        return conv1_f, layer1_f, layer2_f, layer3_f, layer4_f

class ResNetLite(nn.Module):
    def __init__(self, in_ch, block, layers, strides, dilations):
        self.inplanes = 64
        super(ResNetLite, self).__init__()

        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0],
                                       stride=strides[0], dilation=dilations[0])
        self.layer2 = self._make_layer(block, 128, layers[1],
                                       stride=strides[1], dilation=dilations[1])

        self.layer4 = self._make_layer(block, 512, layers[3],
                                       stride=strides[3], dilation=dilations[3])

    def _make_layer(self, block, planes, blocks, stride=1,
                    dilation=1):

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample))
        self.inplanes = planes * block.expansion
        
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        conv1_f = self.relu(x)
        x = self.maxpool(conv1_f)

        layer1_f = self.layer1(x)
        layer2_f = self.layer2(layer1_f)
        layer4_f = self.layer4(layer2_f)
        
        return conv1_f, layer1_f, layer2_f, None, layer4_f

class SkipResnet50(nn.Module):
    def __init__(self, in_ch, concat_channels=64, final_dim=128):
        super(SkipResnet50, self).__init__()
        
        self.concat_channels = concat_channels
        self.final_dim = final_dim
        
        self.image_feature_dim = 256
        self.resnet = ResNetLite(in_ch, Bottleneck, layers=[2, 2, 2, 2], strides=[1, 2, 1, 1],
            dilations=[1, 1, 2, 4])
        
        concat1 = nn.Conv2d(64, concat_channels, kernel_size=3, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(concat_channels)
        relu1 = nn.ReLU(inplace=True)
        
        self.conv1_concat = nn.Sequential(concat1, bn1, relu1)
        
        concat2 = nn.Conv2d(256, concat_channels, kernel_size=3, padding=1, bias=False)
        bn2 = nn.BatchNorm2d(concat_channels)
        relu2 = nn.ReLU(inplace=True)
        up2 = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.res1_concat = nn.Sequential(concat2, bn2, relu2, up2)
        
        concat3 = nn.Conv2d(512, concat_channels, kernel_size=3, padding=1, bias=False)
        bn3 = nn.BatchNorm2d(concat_channels)
        relu3 = nn.ReLU(inplace=True)
        up3 = torch.nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        
        self.res2_concat = nn.Sequential(concat3, bn3, relu3, up3)
        
        concat4 = nn.Conv2d(2048, concat_channels, kernel_size=3, padding=1, bias=False)
        bn4 = nn.BatchNorm2d(concat_channels)
        relu4 = nn.ReLU(inplace=True)
        up4 = torch.nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        
        self.res4_concat = nn.Sequential(concat4, bn4, relu4, up4)
        
        conv_final_1 = nn.Conv2d(4*concat_channels, 128, kernel_size=3, padding=1, stride=2,
            bias=False)
        bn_final_1 = nn.BatchNorm2d(128)
        conv_final_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2, bias=False)
        bn_final_2 = nn.BatchNorm2d(128)
        conv_final_3 = nn.Conv2d(128, final_dim, kernel_size=3, padding=1, bias=False)
        bn_final_3 = nn.BatchNorm2d(final_dim)
        
        self.conv_final = nn.Sequential(conv_final_1, bn_final_1, conv_final_2, bn_final_2,
            conv_final_3, bn_final_3)

    def forward(self, x):
        conv1_f, layer1_f, layer2_f, layer3_f, layer4_f_ori = self.resnet(x)

        conv1_f = self.conv1_concat(conv1_f)
        layer1_f = self.res1_concat(layer1_f)
        layer2_f = self.res2_concat(layer2_f)
        layer4_f = self.res4_concat(layer4_f_ori)
        
        concat_features = torch.cat((conv1_f, layer1_f, layer2_f, layer4_f), dim=1)
        
        final_features = self.conv_final(concat_features)
        
        return concat_features, final_features


class GFE(nn.Module):
    def __init__(self, feats_channels=128):
        super(GFE, self).__init__()
        self.pixel_conv = nn.Conv2d(feats_channels, 16, kernel_size=3, padding=1)
        self.pixel_conv2 = nn.Conv2d(16, 1, kernel_size=1, padding=0)

    def forward(self, feats, beam_size=1):
        batch_size = feats.size(0)
        conv_pixel = self.pixel_conv(feats)
        conv_pixel = F.relu(conv_pixel)
        conv_pixel = self.pixel_conv2(conv_pixel)
        pixel_logits = conv_pixel.view(batch_size, -1)

        logprobs = F.log_softmax(pixel_logits, -1)

        logprob, pred_first = torch.topk(logprobs, beam_size, dim=1)

        return conv_pixel, pixel_logits, pred_first



class GFENet(nn.Module):
    def __init__(self, in_ch):
        super(GFENet, self).__init__()

        self.resnet_skip = SkipResnet50(in_ch)
        self.first_pixel = GFE()

    def forward(self, x):
        concat_features, final_features = self.resnet_skip(x)
        conv_pixel, pixel_logits, pred_first = self.first_pixel(final_features)

        return conv_pixel, pixel_logits


if __name__ == '__main__':
    x = torch.zeros((10, 1, 512, 512))
    f_net = GFENet(1)
    conv_pixel, _ = f_net(x)



