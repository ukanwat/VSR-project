import numpy as np
import megengine
import megengine.module as M
from megengine.module.conv import Conv2d, ConvTranspose2d
import megengine.functional as F
import math
from . import default_init_weights


class h_sigmoid(M.Module):
    def __init__(self):
        super(h_sigmoid, self).__init__()

    def forward(self, x):
        return F.relu6(x + 3) / 6


class h_swish(M.Module):
    def __init__(self):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid()

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(M.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()

        mip = max(16, inp // reduction)
        self.conv1 = M.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)

        self.act = h_swish()
        self.conv_h = M.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = M.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.init_weights()

    def forward(self, x):
        identity = x
        n, c, h, w = x.shape
        x_h = F.mean(x, axis=3, keepdims=True)
        x_w = F.mean(x, axis=2, keepdims=True).transpose(0, 1, 3, 2)
        y = F.concat([x_h, x_w], axis=2)
        y = self.conv1(y)

        y = self.act(y)
        x_h = y[:, :, :h, :]
        x_w = y[:, :, h:, :]
        x_w = x_w.transpose(0, 1, 3, 2)
        a_h = F.sigmoid(self.conv_h(x_h))
        a_w = F.sigmoid(self.conv_w(x_w))
        out = identity * a_w * a_h
        return out

    def init_weights(self):
        default_init_weights(self, scale=0.1)
