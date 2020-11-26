import torch.nn as nn
import torch
import math
from efficientnet_pytorch import EfficientNet as EffNet
from src.utils import *
from src.loss import FocalLoss
from torchvision.ops.boxes import nms as nms_torch
from torchvision import models


def nms(dets, thresh):
    anchors = dets[:, :4]
    scores = dets[:, 4]
    return nms_torch(anchors.cpu(), scores.cpu(), thresh)


class ConvBlock(nn.Module):
    def __init__(self, num_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3,
                      stride=1, padding=1, groups=num_channels),
            nn.Conv2d(num_channels, num_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=num_channels,
                           momentum=0.99, eps=1e-5),
            nn.LeakyReLU(inplace=True))

    def forward(self, input):
        return self.conv(input)


class BiFPN(nn.Module):
    ''' changed
    Please read EffDet
    '''

    def __init__(self, num_channels):
        super(BiFPN, self).__init__()
        # Conv layers
        self.conv5_up = ConvBlock(num_channels)
        self.conv4_up = ConvBlock(num_channels)
        self.conv3_up = ConvBlock(num_channels)
        self.conv4_down = ConvBlock(num_channels)
        self.conv5_down = ConvBlock(num_channels)
        self.conv6_down = ConvBlock(num_channels)

        # Feature scaling layers
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.downsample = nn.MaxPool2d(kernel_size=2)
        self.Softmax = nn.Softmax(dim=0)

        # Weight
        self.p5_w1 = nn.Parameter(torch.ones(2))
        self.p4_w1 = nn.Parameter(torch.ones(2))
        self.p3_w1 = nn.Parameter(torch.ones(2))

        self.p4_w2 = nn.Parameter(torch.ones(3))
        self.p5_w2 = nn.Parameter(torch.ones(3))
        self.p6_w2 = nn.Parameter(torch.ones(2))

    def forward(self, inputs):
        """
            P7_0 -------------------------- P7_2 -------->

            P6_0 ---------- P6_1 ---------- P6_2 -------->

            P5_0 ---------- P5_1 ---------- P5_2 -------->

            P4_0 ---------- P4_1 ---------- P4_2 -------->

            P3_0 -------------------------- P3_2 -------->
        """

        # P3_0, P4_0, P5_0, P6_0 and P7_0
        p3_in, p4_in, p5_in, p6_in = inputs

        p5_w1 = self.Softmax(self.p5_w1)
        weight = p5_w1 / (torch.sum(p5_w1, dim=0))
        p5_up = self.conv5_up(
            weight[0] *
            p5_in +
            weight[1] *
            self.upsample(p6_in))

        p4_w1 = self.Softmax(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0))
        p4_up = self.conv4_up(
            weight[0] *
            p4_in +
            weight[1] *
            self.upsample(p5_up))

        p3_w1 = self.Softmax(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0))
        p3_out = self.conv3_up(
            weight[0] *
            p3_in +
            weight[1] *
            self.upsample(p4_up))

        p4_w2 = self.Softmax(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0))
        p4_out = self.conv4_down(
            weight[0] *
            p4_in +
            weight[1] *
            p4_up +
            weight[2] *
            self.downsample(p3_out))
        p5_w2 = self.Softmax(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0))
        p5_out = self.conv5_down(
            weight[0] *
            p5_in +
            weight[1] *
            p5_up +
            weight[2] *
            self.downsample(p4_out))

        p6_w2 = self.Softmax(self.p6_w2)
        weight = p6_w2 / (torch.sum(p6_w2, dim=0))
        p6_out = self.conv6_down(
            weight[0] *
            p6_in +
            weight[1] *
            self.downsample(p5_out))

        return p3_out, p4_out, p5_out, p6_out


class Regressor(nn.Module):
    def __init__(self, in_channels, num_anchors, num_layers):
        super(Regressor, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1))
            layers.append(nn.LeakyReLU(inplace=True))
        self.layers = nn.Sequential(*layers)
        self.header = nn.Conv2d(
            in_channels,
            num_anchors * 4,
            kernel_size=3,
            stride=1,
            padding=1)

    def forward(self, inputs):
        inputs = self.layers(inputs)
        inputs = self.header(inputs)
        output = inputs.permute(0, 2, 3, 1)
        return output.contiguous().view(output.shape[0], -1, 4)


class Classifier(nn.Module):
    def __init__(self, in_channels, num_anchors, num_classes, num_layers):
        super(Classifier, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1))
            layers.append(nn.LeakyReLU(inplace=True))
        self.layers = nn.Sequential(*layers)
        self.header = nn.Conv2d(
            in_channels,
            num_anchors *
            num_classes,
            kernel_size=3,
            stride=1,
            padding=1)
        self.act = nn.Sigmoid()

    def forward(self, inputs):
        inputs = self.layers(inputs)
        inputs = self.header(inputs)
        inputs = self.act(inputs)
        inputs = inputs.permute(0, 2, 3, 1)
        output = inputs.contiguous().view(
            inputs.shape[0],
            inputs.shape[1],
            inputs.shape[2],
            self.num_anchors,
            self.num_classes)
        return output.contiguous().view(output.shape[0], -1, self.num_classes)


class ResNet(nn.Module):
    ''' EffNet --> ResNet '''

    def __init__(self, ):
        super(ResNet, self).__init__()
        model = models.resnet34(pretrained=True)
        del model.fc
        del model.layer4
        self.model = model

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x1 = x = self.model.relu(x)
        x = self.model.maxpool(x)

        x2 = x = self.model.layer1(x)
        x3 = x = self.model.layer2(x)
        x4 = x = self.model.layer3(x)

        return [x1, x2, x3, x4]


class EfficientDet(nn.Module):
    def __init__(self, num_anchors=9, num_classes=10):
        super(EfficientDet, self).__init__()

        self.num_channels = num_channels = 64

        ''' Use ResNet so lots of difference '''
        self.conv1 = nn.Conv2d(
            64,
            num_channels,
            kernel_size=1,
            stride=1,
            padding=0)
        self.conv2 = nn.Conv2d(
            64,
            num_channels,
            kernel_size=1,
            stride=1,
            padding=0)
        self.conv3 = nn.Conv2d(
            128,
            num_channels,
            kernel_size=1,
            stride=1,
            padding=0)
        self.conv4 = nn.Conv2d(
            256,
            num_channels,
            kernel_size=1,
            stride=1,
            padding=0)

        self.bifpn = nn.Sequential(*[BiFPN(num_channels) for _ in range(3)])

        self.num_classes = num_classes
        self.regressor = Regressor(
            in_channels=num_channels,
            num_anchors=num_anchors,
            num_layers=2)
        self.classifier = Classifier(
            in_channels=num_channels,
            num_anchors=num_anchors,
            num_classes=num_classes,
            num_layers=2)

        self.anchors = Anchors()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.focalLoss = FocalLoss()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        prior = 0.01

        self.classifier.header.weight.data.fill_(0)
        self.classifier.header.bias.data.fill_(
            -math.log((1.0 - prior) / prior))

        self.regressor.header.weight.data.fill_(0)
        self.regressor.header.bias.data.fill_(0)

        self.backbone_net = ResNet()
        self.saved_anchors = None

    def forward(self, inputs):
        if len(inputs) == 2:
            is_training = True
            img_batch, annotations = inputs
        else:
            is_training = False
            img_batch = inputs[0]

        c1, c2, c3, c4 = self.backbone_net(img_batch)
        p1 = self.conv1(c1)
        p2 = self.conv2(c2)
        p3 = self.conv3(c3)
        p4 = self.conv4(c4)

        features = [p1, p2, p3, p4]
        features = self.bifpn(features)

        regression = torch.cat([self.regressor(feature)
                                for feature in features], dim=1)
        classification = torch.cat([self.classifier(feature)
                                    for feature in features], dim=1)

        anchors = self.anchors(img_batch)

        if is_training:
            return self.focalLoss(
                classification,
                regression,
                anchors,
                annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(
                transformed_anchors, img_batch)

            scores = torch.max(classification, dim=2, keepdim=True)[0]

            scores_over_thresh = (scores > 0.15)[0, :, 0]

            if scores_over_thresh.sum() == 0:
                return [
                    torch.zeros(0).cuda(),
                    torch.zeros(0).cuda(),
                    torch.zeros(
                        0,
                        4).cuda()]

            classification = classification[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]

            anchors_nms_idx = nms(torch.cat(
                [transformed_anchors, scores], dim=2)[0, :, :], 0.333)

            nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(
                dim=1)
            return [nms_scores, nms_class,
                    transformed_anchors[0, anchors_nms_idx, :]]
