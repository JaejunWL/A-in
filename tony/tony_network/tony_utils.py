"""
    Utility File
    - should contain functions for networks
"""

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch


class Conv2d_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, \
                                    padding="SAME", dilation=(1,1), bias=True, \
                                    norm="batch", activation="relu", \
                                    deconv=False):
        super(Conv2d_layer, self).__init__()
        self.deconv = deconv
        self.conv2d = nn.Sequential()
        self.padding = padding

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        if isinstance(stride, int):
            stride = [stride, stride]
        if isinstance(dilation, int):
            dilation = [dilation, dilation]

        ''' padding '''
        if deconv:
            padding = tuple(int((current_kernel - 1)/2) for current_kernel in kernel_size)
            out_padding = tuple(0 if current_stride == 1 else 1 for current_stride in stride)
        else:
            if padding == "SAME":
                self.f_pad = int((kernel_size[0]-1) * dilation[0])
                self.t_pad = int((kernel_size[1]-1) * dilation[1])
                self.t_l_pad = int(self.t_pad//2)
                self.t_r_pad = self.t_pad - self.t_l_pad
                self.f_l_pad = int(self.f_pad//2)
                self.f_r_pad = self.f_pad - self.f_l_pad
            elif padding == "VALID":
                padding = 0
            else:
                pass

        ''' convolutional layer '''
        if deconv:
            self.conv2d.add_module("deconv2d", nn.ConvTranspose2d(in_channels, out_channels, \
                                                            (kernel_size[0], kernel_size[1]), \
                                                            stride=stride, padding=padding, output_padding=out_padding, dilation=dilation, bias=bias))
        else:
            self.conv2d.add_module("conv2d", nn.Conv2d(in_channels, out_channels, \
                                                            (kernel_size[0], kernel_size[1]), \
                                                            stride=stride, padding=0, dilation=dilation, bias=bias))
        
        ''' normalization '''
        if norm=="batch":
            self.conv2d.add_module("batch_norm", nn.BatchNorm2d(out_channels))
        	# self.conv2d.add_module('batch_norm', nn.SyncBatchNorm(out_channels))
        elif norm=="group":
            self.conv2d.add_module('group_norm', nn.GroupNorm(num_groups=10, num_channels=out_channels))

        ''' activation '''
        if activation=="relu":
            self.conv1d.add_module("relu", nn.ReLU())
        elif activation=="lrelu":
            self.conv1d.add_module("lrelu", nn.LeakyReLU())
        elif activation=="sigmoid":
            self.conv1d.add_module("sigmoid", nn.Sigmoid())
        elif activation=="softplus":
            self.conv1d.add_module("softplus", nn.Softplus())
        elif activation=="sine":
            self.conv1d.add_module("sine", Sine())
        elif activation=="tanh":
            self.conv1d.add_module("tanh", nn.Tanh())

    def forward(self, input):
        # input shape should be : batch x channel x height x width
        if not self.deconv:
            if self.padding == "SAME":
                input = F.pad(input, (self.t_l_pad, self.t_r_pad, self.f_l_pad, self.f_r_pad))
        output = self.conv2d(input)
        return output


class Conv1d_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, \
                                    padding="SAME", dilation=1, bias=True, \
                                    norm="batch", activation="relu", \
                                    mode="conv"):
        super(Conv1d_layer, self).__init__()
        self.conv1d = nn.Sequential()

        ''' padding '''
        if mode=="deconv":
            padding = int(dilation * (kernel_size-1) / 2)
            out_padding = 0 if stride==1 else 1
        elif mode=="conv" or mode=="resize_conv":
            if padding == "SAME":
                pad = int((kernel_size-1) * dilation)
                l_pad = int(pad//2)
                r_pad = pad - l_pad
                padding_area = (l_pad, r_pad)
            elif padding == "VALID":
                padding_area = (0, 0)
            else:
                pass

        ''' convolutional layer '''
        if mode=="deconv":
            self.conv1d.add_module("deconv1d", nn.ConvTranspose1d(in_channels, out_channels, kernel_size, \
                                                            stride=stride, padding=padding, output_padding=out_padding, dilation=dilation, bias=bias))
        elif mode=="resize_conv" or mode=="conv":
            if mode=="resize_conv":
                self.conv1d.add_module(f"{mode}1d_upsample", nn.Upsample(scale_factor=stride, mode='nearest'))
                # self.conv1d.add_module(f"{mode}1d_upsample", nn.Upsample(scale_factor=stride, mode='linear'))
                stride = 1
            self.conv1d.add_module(f"{mode}1d_pad", nn.ReflectionPad1d(padding_area))
            # self.conv1d.add_module(f"{mode}1d_pad", nn.ReplicationPad1d(padding_area))
            self.conv1d.add_module(f"{mode}1d", nn.Conv1d(in_channels, out_channels, kernel_size, \
                                                            stride=stride, padding=0, dilation=dilation, bias=bias))
        
        ''' normalization '''
        if norm=="batch":
            self.conv1d.add_module("batch_norm", nn.BatchNorm1d(out_channels))
        	# self.conv1d.add_module('batch_norm', nn.SyncBatchNorm(out_channels))
        elif norm=="group":
            self.conv1d.add_module('group_norm', nn.GroupNorm(num_groups=10, num_channels=out_channels))

        ''' activation '''
        if activation=="relu":
            self.conv1d.add_module("relu", nn.ReLU())
        elif activation=="lrelu":
            self.conv1d.add_module("lrelu", nn.LeakyReLU())
        elif activation=="sigmoid":
            self.conv1d.add_module("sigmoid", nn.Sigmoid())
        elif activation=="softplus":
            self.conv1d.add_module("softplus", nn.Softplus())
        elif activation=="sine":
            self.conv1d.add_module("sine", Sine())
        elif activation=="tanh":
            self.conv1d.add_module("tanh", nn.Tanh())

    def forward(self, input):
        # input shape should be : batch x channel x height x width
        output = self.conv1d(input)
        return output


class SELayer(nn.Module):
    def __init__(self, dimension, channel, reduction=4):
        super(SELayer, self).__init__()
        self.dimension = dimension
        if dimension==1:
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
        elif dimension==2:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        if self.dimension==1:
            b, c, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1)
        elif self.dimension==2:
            b, c, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SE_ConvBlock(nn.Module):
    def __init__(self, dimension, in_channels, out_channels, kernel_size, stride=1, padding="SAME", dilation=1, bias=True, norm="batch", activation="relu", last_activation="relu", deconv=False):
        super(SE_ConvBlock, self).__init__()
        if dimension==1:
            self.conv1 = Conv1d_layer(in_channels, in_channels, kernel_size, padding=padding, dilation=dilation, bias=bias, norm=norm, activation=activation)
            self.conv2 = Conv1d_layer(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, norm=norm, activation=last_activation, deconv=deconv)
        elif dimension==2:
            self.conv1 = Conv2d_layer(in_channels, in_channels, kernel_size, padding=padding, dilation=dilation, bias=bias, norm=norm, activation=activation)
            self.conv2 = Conv2d_layer(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, norm=norm, activation=last_activation, deconv=deconv)
        self.se = SELayer(dimension=dimension, channel=in_channels)

    def forward(self, input):
        c1_out = self.conv1(input)
        se_out = self.se(c1_out)
        return self.conv2(se_out)


class Res_ConvBlock(nn.Module):
    def __init__(self, dimension, in_channels, out_channels, kernel_size, stride=1, padding="SAME", dilation=1, bias=True, norm="batch", activation="relu", last_activation="relu", deconv=False):
        super(Res_ConvBlock, self).__init__()
        if dimension==1:
            self.conv1 = Conv1d_layer(in_channels, in_channels, kernel_size, padding=padding, dilation=dilation, bias=bias, norm=norm, activation=activation)
            self.conv2 = Conv1d_layer(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, norm=norm, activation=last_activation, deconv=deconv)
        elif dimension==2:
            self.conv1 = Conv2d_layer(in_channels, in_channels, kernel_size, padding=padding, dilation=dilation, bias=bias, norm=norm, activation=activation)
            self.conv2 = Conv2d_layer(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, norm=norm, activation=last_activation, deconv=deconv)

    def forward(self, input):
        c1_out = self.conv1(input) + input
        c2_out = self.conv2(c1_out)
        return c2_out


class ConvBlock(nn.Module):
    def __init__(self, dimension, layer_num, in_channels, out_channels, kernel_size, stride=1, padding="SAME", dilation=1, bias=True, norm="batch", activation="relu", last_activation="relu", deconv=False):
        super(SE_ConvBlock, self).__init__()
        conv_block = []
        if dimension==1:
            for i in range(layer_num-1):
                conv_block.append(Conv1d_layer(in_channels, in_channels, kernel_size, padding=padding, dilation=dilation, bias=bias, norm=norm, activation=activation))
            conv_block.append(Conv1d_layer(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, norm=norm, activation=last_activation, deconv=deconv))
        elif dimension==2:
            for i in range(layer_num-1):
                conv_block.append(Conv2d_layer(in_channels, in_channels, kernel_size, padding=padding, dilation=dilation, bias=bias, norm=norm, activation=activation))
            conv_block.append(Conv2d_layer(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias, norm=norm, activation=last_activation, deconv=deconv))
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, input):
        return self.conv_block(input)



# Functions used in SIRENnet
class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)
        # return torch.sin(input)


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)



# Feature-wise Linear Modulation
class FiLM(nn.Module):
    def __init__(self, condition_len=2048, feature_len=1024):
        super(FiLM, self).__init__()
        # self.film_fc = nn.Linear(condition_len, feature_len*2)
        self.film_fc = nn.Sequential(
            nn.Linear(condition_len, condition_len),
            nn.Sigmoid(),
            nn.Linear(condition_len, feature_len*2)
        )
        self.feat_len = feature_len

    def forward(self, feature, condition):
        film_factor = self.film_fc(condition).unsqueeze(-1)
        r, b = torch.split(film_factor, self.feat_len, dim=1)
        return r*feature + b


class Linear(nn.Module):
    """
    Wrapper class of torch.nn.Linear
    Weight initialize by xavier initialization and bias initialize to zeros.
    """
    def __init__(self, in_features, out_features, bias = True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)
    def forward(self, x):
        return self.linear(x)