import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvReg(nn.Module):
    """Convolutional regression"""

    def __init__(self, s_shape, t_shape, use_relu=True, use_bn=True):
        super(ConvReg, self).__init__()
        self.use_relu = use_relu
        # 决定是否在前向传播中使用Batch normalization
        self.use_bn = use_bn
        s_N, s_C, s_H, s_W = s_shape
        t_N, t_C, t_H, t_W = t_shape
        if s_H == 2 * t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=3, stride=2, padding=1)
        elif s_H * 2 == t_H:
            self.conv = nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1)
        elif s_H >= t_H:
            self.conv = nn.Conv2d(s_C, t_C, kernel_size=(1 + s_H - t_H, 1 + s_W - t_W))
        else:
            raise NotImplemented("student size {}, teacher size {}".format(s_H, t_H))
        
        self.bn = nn.BatchNorm2d(t_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_relu:
            return self.relu(self.bn(x))
        # 判断是否需要bn（这看起来无用）
        elif self.use_bn:
            return self.bn(x)
        else:
            return self.bn(x)


def get_feat_shapes(student, teacher, input_size):
    data = torch.randn(1, 3, *input_size)
    with torch.no_grad():
        _, feat_s = student(data)
        _, feat_t = teacher(data)
    feat_s_shapes = [f.shape for f in feat_s["feats"]]
    feat_t_shapes = [f.shape for f in feat_t["feats"]]
    return feat_s_shapes, feat_t_shapes

# by CDK  服务于WKD方法
class ConvRegBottleNeck(nn.Module):
    def __init__(self, s_shape, t_shape, c_hidden, use_relu=True, use_bn=True):
        super(ConvRegBottleNeck, self).__init__()
        self.use_relu = use_relu
        self.use_bn = use_bn
        s_N, s_C, s_H, s_W = s_shape
        t_N, t_C, t_H, t_W = t_shape

        self.conv1 = nn.Conv2d(s_C, c_hidden, kernel_size=1)

        if s_H * 2 == t_H:
            self.conv2 = nn.ModuleList([nn.ConvTranspose2d(s_C, t_C, kernel_size=4, stride=2, padding=1),
                                        nn.Conv2d(t_C, t_C, kernel_size=3, stride=1, padding=1)])
        elif s_H == t_H:
            self.conv2 = nn.Conv2d(c_hidden, c_hidden, kernel_size=3, stride=1, padding=1)
        else:
            raise NotImplemented("student size {}, teacher size {}".format(s_H, t_H))

        self.conv3 = nn.Conv2d(c_hidden, t_C, kernel_size=1)

        self.bn = nn.BatchNorm2d(t_C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.use_relu:
            return self.relu(self.bn(x))
        elif self.use_bn:
            return self.bn(x)
        else:
            return x

