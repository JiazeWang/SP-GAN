# encoding=utf-8

import numpy as np
import math
import sys
import os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

from collections import namedtuple
cudnn.benchnark=True
from torch.nn import AvgPool2d, Conv1d, Conv2d, ConvTranspose2d, Embedding, LeakyReLU, Module

neg = [1e-2, 0.2][0]

class BasicConv1D(nn.Module):
    def __init__(self, Fin, Fout, act=True, norm="BN", kernal=1):
        super(BasicConv1D, self).__init__()

        self.conv = nn.Conv1d(Fin,Fout,kernal)
        if act:
            self.act = nn.LeakyReLU(inplace=True)
        else:
            self.act = None

        if norm is not None:
            self.norm = nn.BatchNorm1d(Fout) if norm=="BN" else nn.InstanceNorm1d(Fout)
        else:
            self.norm = None

    def forward(self, x):
        x = self.conv(x)  # Bx2CxNxk

        if self.norm is not None:
            x = self.norm(x)

        if self.act is not None:
            x = self.act(x)

        return x


class Generator(nn.Module):
    def __init__(self, opts, num_point=2048):
        super(Generator, self).__init__()
        self.num_point = num_point
        self.opts = opts
        BN = True
        self.small_d = opts.small_d

        self.mlps = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(neg, inplace=True),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(neg, inplace=True),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(neg, inplace=True),
        )

        self.mode = ["max","max_avg"][0]

        if self.mode == "max":
            dim = 1024
        else:
            dim = 512

        if self.small_d:
            dim = dim//2

        self.fc2 = nn.Sequential(
            nn.Conv1d(256,dim,1),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(neg, inplace=True)
        )


        self.mlp = nn.Sequential(
            nn.Conv1d(dim+1024, 512),
            nn.LeakyReLU(neg, inplace=True),
            #nn.Dropout(0.5),
            nn.Conv1d(512, 256),
            nn.LeakyReLU(neg, inplace=True),
            #nn.Dropout(0.5),
            nn.Conv1d(256, 64),
            nn.LeakyReLU(neg, inplace=True),
            #nn.Dropout(0.5),
            nn.Conv1d(64, 3)
            )

    def forward(self, x):
        B,N,_ = x.size()
        x = x.transpose(1, 2)
        if self.opts.z_norm:
            x = x / (x.norm(p=2, dim=-1, keepdim=True)+1e-8)
        x = self.mlps(x)
        x = self.fc2(x)

        x_feat = torch.max(x, 2, keepdim=True)[0]
        print("x_feat", x_feat.shape)
        x_feat = x_feat.view(-1, 1024, 1).repeat(1, 1, self.num_point)
        print("x_feat_new", x_feat.shape)
        print("x:", x.shape)
        x = torch.cat([x, x_feat], 1)
        x = self.mlp(x)
        x = x.transpose(1, 2)

        return x
