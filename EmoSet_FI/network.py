import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.autograd import Variable
import math
import torch.nn.utils.weight_norm as weightNorm
from collections import OrderedDict
# from TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
# from TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

vgg_dict = {"vgg11":models.vgg11, "vgg13":models.vgg13, "vgg16":models.vgg16, "vgg19":models.vgg19, 
"vgg11bn":models.vgg11_bn, "vgg13bn":models.vgg13_bn, "vgg16bn":models.vgg16_bn, "vgg19bn":models.vgg19_bn} 
class VGGBase(nn.Module):
  def __init__(self, vgg_name):
    super(VGGBase, self).__init__()
    model_vgg = vgg_dict[vgg_name](pretrained=True)
    self.features = model_vgg.features
    self.classifier = nn.Sequential()
    for i in range(6):
        self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])
    self.in_features = model_vgg.classifier[6].in_features

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x

res_dict = {"resnet18":models.resnet18, "resnet34":models.resnet34, "resnet50":models.resnet50, 
"resnet101":models.resnet101, "resnet152":models.resnet152, "resnext50":models.resnext50_32x4d, "resnext101":models.resnext101_32x8d}

class ResBase(nn.Module):
    def __init__(self, res_name):
        super(ResBase, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class ResBase_BYOT(nn.Module):
    def __init__(self, res_name):
        super(ResBase_BYOT, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)


        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


# class ViT(nn.Module):
#     def __init__(self):
#         super(ViT, self).__init__()
#         config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
#         config_vit.n_classes = 8
#         config_vit.n_skip = 3
#         config_vit.patches.grid = (int(224 / 16), int(224 / 16))
#         self.feature_extractor = ViT_seg(config_vit, img_size=[224, 224], num_classes=config_vit.n_classes)
#         self.feature_extractor.load_from(weights=np.load(config_vit.pretrained_path))
#         self.in_features = 2048
#
#     def forward(self, x):
#         _, feat = self.feature_extractor(x)
#         return feat

import timm
class ViT(nn.Module):
    def __init__(self, model_name='vit_base_patch16_224_in21k', pretrained=True):
        # Load the ViT model
        # self.vit_model = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        super(ViT, self).__init__()
        self.vit_model = timm.create_model(model_name, pretrained=False)
        # print(self.vit_model)
        state_dict = torch.load("/nfs/ofs-902-1/object-detection/zhujiankun/EDA/code/TransDA/model/vit_hf/imagenet21k/pytorch_model.bin")
        self.vit_model.load_state_dict(state_dict, strict=False)
        self.vit_model = torch.nn.Sequential(*(list(self.vit_model.children())[:-1]))

        self.in_features = 768

    def forward(self, img):
        # Extract features using timm's features_only mode
        features = self.vit_model(img)
        # Take the last feature if multiple stages are returned
        if isinstance(features, (list, tuple)):
            features = features[-1]
        features = features[:,0,:]
        return features


class ResBase_BYOT(nn.Module):
    def __init__(self, res_name):
        super(ResBase_BYOT, self).__init__()
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

        # 为每个layer定义独有的Bottleneck层、Pooling层和全连接层
        # self.bottlenecks = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv2d(256, 256, kernel_size=1, bias=False),
        #         nn.BatchNorm2d(256),
        #         nn.ReLU(inplace=True)
        #     ),
        #     nn.Sequential(
        #         nn.Conv2d(512, 512, kernel_size=1, bias=False),
        #         nn.BatchNorm2d(512),
        #         nn.ReLU(inplace=True)
        #     ),
        #     nn.Sequential(
        #         nn.Conv2d(1024, 1024, kernel_size=1, bias=False),
        #         nn.BatchNorm2d(1024),
        #         nn.ReLU(inplace=True)
        #     ),
        #     nn.Sequential(
        #         nn.Conv2d(2048, 2048, kernel_size=1, bias=False),
        #         nn.BatchNorm2d(2048),
        #         nn.ReLU(inplace=True)
        #     )
        # ])
        self.bottlenecks = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(2048),
                nn.ReLU(inplace=True)
            ) for _ in range(4)
        ])
        self.adjust_channels = nn.ModuleList([
            nn.Conv2d(256, 2048, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(512, 2048, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Conv2d(1024, 2048, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Identity()  # 对于最后一层不需要调整
        ])
        # self.pools = nn.ModuleList([
        #     nn.AdaptiveAvgPool2d((1, 1)) for _ in range(4)
        # ])
        # 为前三个layer定义自适应平均池化层
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.AdaptiveAvgPool2d((1, 1)),
            # 第四个layer层不需要额外的pooling，因为已经有avgpool
        ])

        # self.fcs = nn.ModuleList([
        #     nn.Linear(256, 8),
        #     nn.Linear(512, 8),
        #     nn.Linear(1024, 8),
        #     nn.Linear(2048, 8)
        # ])
        self.fcs = nn.ModuleList([
            nn.Linear(2048, 8) for _ in range(4)
        ])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x1 = self.adjust_channels[0](x)
        x_bottleneck_1 = self.bottlenecks[0](x1)
        x_pool_1 = self.pools[0](x_bottleneck_1)
        middle1_fea = x_pool_1
        x_flat_1 = torch.flatten(x_pool_1, 1)
        x_out_1 = self.fcs[0](x_flat_1)

        x = self.layer2(x)
        x2 = self.adjust_channels[1](x)
        x_bottleneck_2 = self.bottlenecks[1](x2)
        x_pool_2 = self.pools[1](x_bottleneck_2)
        middle2_fea = x_pool_2
        x_flat_2 = torch.flatten(x_pool_2, 1)
        x_out_2 = self.fcs[1](x_flat_2)


        x = self.layer3(x)
        x3 = self.adjust_channels[2](x)
        x_bottleneck_3 = self.bottlenecks[2](x3)
        x_pool_3 = self.pools[2](x_bottleneck_3)
        middle3_fea = x_pool_3
        x_flat_3 = torch.flatten(x_pool_3, 1)
        x_out_3 = self.fcs[2](x_flat_3)

        x = self.layer4(x)
        x = self.avgpool(x)
        final_fea = x
        x = x.view(x.size(0), -1)
        return x, x_out_1, x_out_2, x_out_3, final_fea, middle1_fea, middle2_fea, middle3_fea



class feat_bottleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super(feat_bottleneck, self).__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x

class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight")
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x

class feat_classifier_two(nn.Module):
    def __init__(self, class_num, input_dim, bottleneck_dim=256):
        super(feat_classifier_two, self).__init__()
        self.type = type
        self.fc0 = nn.Linear(input_dim, bottleneck_dim)
        self.fc0.apply(init_weights)
        self.fc1 = nn.Linear(bottleneck_dim, class_num)
        self.fc1.apply(init_weights)

    def forward(self, x):
        x = self.fc0(x)
        x = self.fc1(x)
        return x

class Res50(nn.Module):
    def __init__(self):
        super(Res50, self).__init__()
        model_resnet = models.resnet50(pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features
        self.fc = model_resnet.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return x, y