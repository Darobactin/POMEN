import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from thop import profile
from functools import reduce


def swish(x):
    return x * x.sigmoid()

def drop_connect_2d(x, drop_ratio):
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x

def drop_connect_3d(x, drop_ratio):
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x

class SE2D(nn.Module):
    '''Squeeze-and-Excitation block with Swish.'''

    def __init__(self, in_channels, se_channels):
        super(SE2D, self).__init__()
        self.se1 = nn.Conv2d(in_channels, se_channels,
                             kernel_size=1, bias=True)
        self.se2 = nn.Conv2d(se_channels, in_channels,
                             kernel_size=1, bias=True)

    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, (1, 1))
        out = swish(self.se1(out))
        out = self.se2(out).sigmoid()
        out = x * out
        return out


class SE3D(nn.Module):
    '''Squeeze-and-Excitation block with Swish.'''

    def __init__(self, in_channels, se_channels):
        super(SE3D, self).__init__()
        self.se1 = nn.Conv3d(in_channels, se_channels,
                             kernel_size=1, bias=True)
        self.se2 = nn.Conv3d(se_channels, in_channels,
                             kernel_size=1, bias=True)

    def forward(self, x):
        out = F.adaptive_avg_pool3d(x, (1, 1, 1))
        out = swish(self.se1(out))
        out = self.se2(out).sigmoid()
        out = x * out
        return out


class Block2D(nn.Module):
    '''expansion + depthwise + pointwise + squeeze-excitation'''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 expand_ratio=1,
                 se_ratio=0.,
                 drop_rate=0.):
        super(Block2D, self).__init__()
        self.stride = stride
        self.drop_rate = drop_rate
        self.expand_ratio = expand_ratio

        # Expansion
        channels = expand_ratio * in_channels
        self.conv1 = nn.Conv2d(in_channels,
                               channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        # Depthwise conv
        self.conv2 = nn.Conv2d(channels,
                               channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=(1 if kernel_size == 3 else 2),
                               groups=channels,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        # SE layers
        se_channels = int(in_channels * se_ratio)
        self.se = SE2D(channels, se_channels)

        # Output
        self.conv3 = nn.Conv2d(channels,
                               out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Skip connection if in and out shapes are the same (MV-V2 style)
        self.has_skip = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        out = x if self.expand_ratio == 1 else swish(self.bn1(self.conv1(x)))
        out = swish(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))
        if self.has_skip:
            if self.training and self.drop_rate > 0:
                out = drop_connect_2d(out, self.drop_rate)
            out = out + x
        return out


class Block3D(nn.Module):
    '''expansion + depthwise + pointwise + squeeze-excitation'''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 expand_ratio=1,
                 se_ratio=0.,
                 drop_rate=0.):
        super(Block3D, self).__init__()
        self.stride = stride
        self.drop_rate = drop_rate
        self.expand_ratio = expand_ratio

        # Expansion
        channels = expand_ratio * in_channels
        self.conv1 = nn.Conv3d(in_channels,
                               channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm3d(channels)

        # Depthwise conv
        self.conv2 = nn.Conv3d(channels,
                               channels,
                               kernel_size=kernel_size,
                               stride=stride,
                               padding=(1 if kernel_size == 3 else 2),
                               groups=channels,
                               bias=False)
        self.bn2 = nn.BatchNorm3d(channels)

        # SE layers
        se_channels = int(in_channels * se_ratio)
        self.se = SE3D(channels, se_channels)

        # Output
        self.conv3 = nn.Conv3d(channels,
                               out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)

        # Skip connection if in and out shapes are the same (MV-V2 style)
        self.has_skip = (stride == 1) and (in_channels == out_channels)

    def forward(self, x):
        out = x if self.expand_ratio == 1 else swish(self.bn1(self.conv1(x)))
        out = swish(self.bn2(self.conv2(out)))
        out = self.se(out)        
        out = self.bn3(self.conv3(out))
        
        if self.has_skip:
            if self.training and self.drop_rate > 0:
                out = drop_connect_3d(out, self.drop_rate)
            out = out + x       
        return out


class DualVectorFoil(nn.Module):
    def __init__(self, cube_size, depth_dim, in_channels=1):
        super(DualVectorFoil, self).__init__()
        self.cube_size = list(cube_size)
        self.feature_order = [i for i in range(5)]
        depth_dim += 2
        if depth_dim != 2:
            self.feature_order[2] = depth_dim
            self.feature_order[depth_dim] = 2
        self.depth_dim = depth_dim - 2
        if self.depth_dim != 0:
            self.cube_size[0] = self.cube_size[0] + self.cube_size[self.depth_dim]
            self.cube_size[self.depth_dim] = self.cube_size[0] - self.cube_size[self.depth_dim]
            self.cube_size[0] = self.cube_size[0] - self.cube_size[self.depth_dim]

        self.in_channels = in_channels
        #################################################
        self.out_channels = self.in_channels * 1  # 8x  
        #################################################
        self.conv1 = nn.Conv3d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(self.out_channels)
        
        self.layers = self._make_layers(kernel_list=(5, 3), pooling=True)
        self.in_channels = self.out_channels

        self.se = SE2D(self.cube_size[0], max(int(self.cube_size[0] * 0.25), 2))
        self.bn2 = nn.BatchNorm2d(self.out_channels)
    
    def _make_layers(self, kernel_list, pooling=False):
        layers = []
        layer_count = 0
        for kernel_size in kernel_list:
            self.in_channels = self.out_channels
            self.out_channels *= kernel_size - 1
            padding_size = (kernel_size - 1) // 2
            if layer_count == 0:
                layers.append(nn.Conv3d(self.in_channels, self.out_channels, kernel_size=kernel_size, stride=2, padding=padding_size, bias=True))
                self.cube_size = [math.ceil(self.cube_size[i] / 2) for i in range(3)]
            else:
                layers.append(nn.Conv3d(self.in_channels, self.out_channels, kernel_size=kernel_size, stride=1, padding=padding_size, bias=True))
            if pooling:
                self.cube_size[0] = max(self.cube_size[0] // 2, 1)
                layers.append(nn.AdaptiveAvgPool3d(output_size=self.cube_size))
            layers.append(nn.BatchNorm3d(self.out_channels))
            layer_count += 1

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(self.feature_order).contiguous()
        out = swish(self.bn1(self.conv1(x)))
        out = swish(self.layers(out))
        
        batch_size, C, D, H, W = out.size()
        out = out.view(batch_size * C, D, H, W)
        out = self.se(out)
        out = out.view(batch_size, C, D, H, W)
        out = self.bn2(F.adaptive_avg_pool3d(out, (1, H, W)).squeeze(2))
        return out    


class LinearTransform2D(nn.Module):
    def __init__(self, in_channels, feature_length, dropout_rate=0):
        super(LinearTransform2D, self).__init__()
        self.feature_length = feature_length
        self.dropout_rate = dropout_rate
        self.conv1x1 = nn.Conv2d(in_channels, feature_length, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        out = self.conv1x1(x)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        if self.training and self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate)
        return out


class LinearTransform3D(nn.Module):
    def __init__(self, in_channels, feature_length, dropout_rate=0):
        super(LinearTransform3D, self).__init__()
        self.feature_length = feature_length
        self.dropout_rate = dropout_rate
        self.conv1x1 = nn.Conv3d(in_channels, feature_length, kernel_size=1, stride=1, bias=True)

    def forward(self, x):
        out = self.conv1x1(x)
        out = F.adaptive_avg_pool3d(out, 1)
        out = out.view(out.size(0), -1)
        if self.training and self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate)
        return out


class HomoGatedFusion3(nn.Module):
    def __init__(self, in_length, feature_length):
        super(HomoGatedFusion3, self).__init__()
        self.in_1 = nn.Linear(in_length, feature_length)
        self.in_2 = nn.Linear(in_length, feature_length)
        self.in_3 = nn.Linear(in_length, feature_length)

        self.mlp1_1 = nn.Linear(feature_length, feature_length * 3)
        self.mlp1_2 = nn.Linear(feature_length, feature_length * 3)
        self.mlp1_3 = nn.Linear(feature_length, feature_length * 3)
        self.mlp1_4 = nn.Linear(feature_length * 3, feature_length)

        self.mlp2_1 = nn.Linear(feature_length, feature_length * 3)
        self.mlp2_2 = nn.Linear(feature_length, feature_length * 3)
        self.mlp2_3 = nn.Linear(feature_length, feature_length * 3)
        self.mlp2_4 = nn.Linear(feature_length * 3, feature_length)

        self.mlp3_1 = nn.Linear(feature_length, feature_length * 3)
        self.mlp3_2 = nn.Linear(feature_length, feature_length * 3)
        self.mlp3_3 = nn.Linear(feature_length, feature_length * 3)
        self.mlp3_4 = nn.Linear(feature_length * 3, feature_length)
        
    def forward(self, x1, x2, x3):
        x1 = F.relu(self.in_1(x1), inplace=True)
        x2 = F.relu(self.in_2(x2), inplace=True)
        x3 = F.relu(self.in_3(x3), inplace=True)

        out1_1 = self.mlp1_1(x1)
        out1_2 = self.mlp1_2(x2)
        out1_3 = self.mlp1_3(x3)
        out1_4 = out1_1 + out1_2 + out1_3
        # out1_4 = out1_4.sigmoid()
        out1_5 = self.mlp1_4(out1_4)
        out1 = torch.mul(x1, out1_5.sigmoid())

        out2_1 = self.mlp2_1(x1)
        out2_2 = self.mlp2_2(x2)
        out2_3 = self.mlp2_3(x3)
        out2_4 = out2_1 + out2_2 + out2_3
        # out2_4 = out2_4.sigmoid()
        out2_5 = self.mlp2_4(out2_4)
        out2 = torch.mul(x2, out2_5.sigmoid())

        out3_1 = self.mlp3_1(x1)
        out3_2 = self.mlp3_2(x2)
        out3_3 = self.mlp3_3(x3)
        out3_4 = out3_1 + out3_2 + out3_3
        # out3_4 = out3_4.sigmoid()
        out3_5 = self.mlp3_4(out3_4)
        out3 = torch.mul(x3, out3_5.sigmoid())

        out = out1 + out2 + out3        
        return out


class POMEN(nn.Module):
    def __init__(self, cube_size, cfg, num_classes=10, in_channels=1):
        super(POMEN, self).__init__()
        self.cfg = cfg
        self.cube_size = cube_size
        self.out_channels = in_channels
        if len(cfg['out_channels_3d']) > 0:
            self.out_channels *= 12
            self.conv1_3d = nn.Conv3d(in_channels,
                                self.out_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
            self.bn1_3d = nn.BatchNorm3d(self.out_channels)
            self.layers_3d = self._make_layers_3d(in_channels=self.out_channels)

            self.out_channels = cfg['out_channels_3d'][-1]
            reduce_factor = reduce(lambda x, y:x * y, cfg['stride_3d'])
            self.cube_size = [math.ceil(self.cube_size[i] / reduce_factor) for i in range(len(self.cube_size))]
        if len(cfg['out_channels_2d']) > 0:
            self.proj_xy = DualVectorFoil(self.cube_size, depth_dim=2, in_channels=self.out_channels)
            self.proj_xz = DualVectorFoil(self.cube_size, depth_dim=1, in_channels=self.out_channels)
            self.proj_yz = DualVectorFoil(self.cube_size, depth_dim=0, in_channels=self.out_channels)

            self.out_channels *= 8
            self.stream_xy = self._make_layers_2d(self.out_channels)
            self.stream_yz = self._make_layers_2d(self.out_channels)
            self.stream_xz = self._make_layers_2d(self.out_channels)

            self.out_channels = cfg['out_channels_2d'][-1]
            self.linear_xy = LinearTransform2D(self.out_channels, cfg['fc_channels'], cfg['dropout_rate'])
            self.linear_yz = LinearTransform2D(self.out_channels, cfg['fc_channels'], cfg['dropout_rate'])
            self.linear_xz = LinearTransform2D(self.out_channels, cfg['fc_channels'], cfg['dropout_rate'])

            self.out_channels = 128
            self.concat = HomoGatedFusion3(cfg['fc_channels'], self.out_channels)
        else:
            self.out_channels = cfg['out_channels_3d'][-1]
            self.linear_xyz = LinearTransform3D(self.out_channels, cfg['fc_channels'], cfg['dropout_rate'])

            self.out_channels = 128
            self.squeeze = nn.Linear(cfg['fc_channels'], self.out_channels)

        self.linear = nn.Linear(self.out_channels, num_classes)
        # self.linear = nn.Linear(cfg['fc_channels'], num_classes)

    def _make_layers_2d(self, in_channels):
        layers = []
        cfg = [self.cfg[k] for k in ['expansion_2d', 'out_channels_2d', 'num_blocks_2d',
                                     'kernel_size_2d', 'stride_2d']]
        b = 0  # decide if drop connect happens
        blocks = sum(self.cfg['num_blocks_2d'])
        for expansion, out_channels, num_blocks, kernel_size, stride in zip(*cfg):
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                drop_rate = self.cfg['drop_connect_rate'] * b / blocks
                layers.append(
                    Block2D(in_channels,
                          out_channels,
                          kernel_size,
                          stride,
                          expansion,
                          se_ratio=0.25,
                          drop_rate=drop_rate))
                in_channels = out_channels
        return nn.Sequential(*layers)
    
    def _make_layers_3d(self, in_channels):
        layers = []
        cfg = [self.cfg[k] for k in ['expansion_3d', 'out_channels_3d', 'num_blocks_3d',
                                     'kernel_size_3d', 'stride_3d']]
        b = 0  # decide if drop connect happens
        blocks = sum(self.cfg['num_blocks_3d'])
        for expansion, out_channels, num_blocks, kernel_size, stride in zip(*cfg):
            strides = [stride] + [1] * (num_blocks - 1)
            for stride in strides:
                drop_rate = self.cfg['drop_connect_rate'] * b / blocks
                layers.append(
                    Block3D(in_channels,
                          out_channels,
                          kernel_size,
                          stride,
                          expansion,
                          se_ratio=0.25,
                          drop_rate=drop_rate))
                in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        if len(self.cfg['out_channels_3d']) > 0:
            out_3d = swish(self.bn1_3d(self.conv1_3d(x)))
            out_3d = self.layers_3d(out_3d)
        else:
            out_3d = x
        if len(self.cfg['out_channels_2d']) > 0:
            in_xy = self.proj_xy(out_3d)
            in_yz = self.proj_yz(out_3d)
            in_xz = self.proj_xz(out_3d)

            in_xy = self.stream_xy(in_xy)
            in_yz = self.stream_yz(in_yz)
            in_xz = self.stream_xz(in_xz)

            out_xy = self.linear_xy(in_xy)
            out_yz = self.linear_yz(in_yz)
            out_xz = self.linear_xz(in_xz)
            out = self.concat(out_xy, out_yz, out_xz)
        else:
            out = self.linear_xyz(out_3d)
            out = F.relu(self.squeeze(out))
        out = self.linear(out)
        return out


def POMENS1(cube_size, num_classes=6, in_channels=1):
    cfg = {
        'num_blocks_3d': [1],
        'expansion_3d': [1],
        'out_channels_3d': [6],
        'kernel_size_3d': [3],
        'stride_3d': [1],
        'num_blocks_2d': [2, 3, 3, 4, 1],
        'expansion_2d': [6, 6, 6, 6, 6],
        'out_channels_2d': [80, 128, 256, 384, 640],
        'kernel_size_2d': [5, 3, 5, 5, 3],
        'stride_2d': [2, 2, 1, 2, 1],
        'fc_channels': 1280,
        'dropout_rate': 0.2,
        'drop_connect_rate': 0.2,
    }
    return POMEN(cube_size, cfg, num_classes=num_classes, in_channels=in_channels)


def POMENS2(cube_size, num_classes=6, in_channels=1):
    cfg = {
        'num_blocks_3d': [1, 2],
        'expansion_3d': [1, 6],
        'out_channels_3d': [6, 16],
        'kernel_size_3d': [3, 3],
        'stride_3d': [1, 2],
        'num_blocks_2d': [3, 3, 4, 1],
        'expansion_2d': [6, 6, 6, 6],
        'out_channels_2d': [224, 448, 624, 1056],
        'kernel_size_2d': [3, 5, 5, 3],
        'stride_2d': [2, 1, 2, 1],
        'fc_channels': 2112,
        'dropout_rate': 0.2,
        'drop_connect_rate': 0.2,
    }
    return POMEN(cube_size, cfg, num_classes=num_classes, in_channels=in_channels)


def POMENS3(cube_size, num_classes=6, in_channels=1):
    cfg = {
        'num_blocks_3d': [1, 2, 2],
        'expansion_3d': [1, 6, 6],
        'out_channels_3d': [6, 16, 54],
        'kernel_size_3d': [3, 3, 5],
        'stride_3d': [1, 2, 2],
        'num_blocks_2d': [3, 4, 1],
        'expansion_2d': [6, 6, 6],
        'out_channels_2d': [864, 1200, 2048],
        'kernel_size_2d': [5, 5, 3],
        'stride_2d': [1, 2, 1],
        'fc_channels': 4096,
        'dropout_rate': 0.2,
        'drop_connect_rate': 0.2,
    }
    return POMEN(cube_size, cfg, num_classes=num_classes, in_channels=in_channels)


def POMENS4(cube_size, num_classes=6, in_channels=1):
    cfg = {
        'num_blocks_3d': [1, 2, 2, 3],
        'expansion_3d': [1, 6, 6, 6],
        'out_channels_3d': [6, 16, 54, 214],
        'kernel_size_3d': [3, 3, 5, 3],
        'stride_3d': [1, 2, 2, 2],
        'num_blocks_2d': [4, 1],
        'expansion_2d': [6, 6],
        'out_channels_2d': [2400, 4096],
        'kernel_size_2d': [5, 3],
        'stride_2d': [2, 1],
        'fc_channels': 8192,
        'dropout_rate': 0.2,
        'drop_connect_rate': 0.2,
    }
    return POMEN(cube_size, cfg, num_classes=num_classes, in_channels=in_channels)


def test():
    net = POMENS2((50, 64, 64), num_classes=6)
    x = torch.randn(1, 1, 50, 64, 64)
    # y = net(x)
    # print(y.shape)
    summary(net.cuda(), (1, 1, 50, 64, 64))
    flops, params = profile(net, inputs=(x.to('cuda'), ))
    # flops, params = profile(net, inputs=(x, ))
    print(flops, params)


if __name__ == '__main__':
    test()
