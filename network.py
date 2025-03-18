#!/usr/bin/env python
# coding: utf-8

"""
Neural Network Architecture for Traffic Sign Classification
-------------------------------------------------
ResNet18 model implementation for traffic sign classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=5):  # 5 classes for traffic signs
        super(ResNet, self).__init__()
        self.in_planes = 64

        # TODO: Verify the input image size and adjust the network structure accordingly
        # For 32x32 images, use a smaller initial convolution and different pooling strategy
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # TODO: You can uncomment the print statements below to visualize the output dimensions at each layer,
        # which will help you understand the network structure
        
        #print(f"Input shape: {x.shape}")
        out = F.relu(self.bn1(self.conv1(x)))
        #print(f"After initial conv: {out.shape}")
        out = self.layer1(out)
        #print(f"After layer1: {out.shape}")
        out = self.layer2(out)
        #print(f"After layer2: {out.shape}")
        out = self.layer3(out)
        #print(f"After layer3: {out.shape}")
        out = self.layer4(out)
        #print(f"After layer4: {out.shape}")
        
        # Apply average pooling
        # The pooling size should be adjusted based on the input image size
        out = F.avg_pool2d(out, 4)  # For 32x32 input
        #print(f"After avg_pool2d: {out.shape}")
        
        out = out.view(out.size(0), -1)
        #print(f"After flatten: {out.shape}")
        out = self.linear(out)
        #print(f"Output shape: {out.shape}")
        return out


def ResNet18(num_classes=5):
    """
    Creates a ResNet18 model with specified number of output classes
    
    Args:
        num_classes (int): Number of output classes
        
    Returns:
        ResNet: ResNet18 model
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)