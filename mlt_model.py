import torch
from torch import nn
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
sigmoid = nn.Sigmoid()
# helpers

#model 3
class Convd(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.Convv = nn.Sequential(
            nn.Conv2d(channels,channels, kernel_size=(3,3) ,padding=1),
            nn.ReLU(),
            nn.Conv2d(channels,channels, kernel_size=(3,3) ,padding=1),
            nn.ReLU()
        )
        self.finnal = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=(5,5), padding=2)
        )
    def forward(self, x):
        x1 = self.Convv(x)
        x = x1 + x
        return self.finnal(x)

#model 4
class Conv_Res_LOS(nn.Module):
    def __init__(self,in_channels):
        super(Conv_Res_LOS, self).__init__()
        self.Convv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, in_channels, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )
        self.finalos = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=(5, 5), padding=2)
        )
        self.sfmx = nn.Softmax(dim=1)
    def forward(self,x):
        x1 = self.Convv(x)
        x1 = x1 + x
        x1 = self.Convv(x1)
        x1 = x1 + x
        y = - torch.log(self.sfmx(self.finalos(x1)))
        return y  # b,3,200,200

class Conv_Res(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.Convv = nn.Sequential(
            nn.Conv2d(in_channels,32, kernel_size=(3,3) ,padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(32,in_channels, kernel_size=(3,3) ,padding=1),
            nn.ReLU()
        )
        self.finnal = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(5,5), padding=2)
        )
        self.finalos = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=(5, 5), padding=2)
        )
        self.sfmx = nn.Softmax(dim=1)
    def forward(self, x, args):
        x1 = self.Convv(x)
        x1 = x1 + x
        x1 = self.Convv(x1)
        x1 = x1 + x

        if  args.state == 0:
            y = self.finnal(x1)
        elif args.state == 1:
            y1 = self.finnal(x1)
            y2 = - torch.log(self.sfmx(self.finalos(x1)))
            y = (y1,y2)
        elif args.state == 2:
            y = - torch.log(self.sfmx(self.finalos(x1)))
        else:
            raise KeyError
        return y

# model 5
class mltask(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(mltask,self).__init__()
        self.net1 = net1(in_channels,out_channels)
        self.net2 = net2(out_channels)
        # self.net2 = Conv_Res_LOS(in_channels)
    def forward(self, x, args):
        y = self.net1(x)
        if args.period == 1:
            y = self.net2(y)
        else :
            y = self.net2(y.detach())
        return y

class net1(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(net1, self).__init__()
        self.Convv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, in_channels, kernel_size=(3, 3), padding=1),
            nn.ReLU()
        )
        self.final = nn.Conv2d(in_channels, out_channels,kernel_size=(3,3),padding=1)
    def forward(self,x):
        x1 = self.Convv(x)
        x1 = x1 + x
        x1 = self.Convv(x1)
        x1 = x1 + x
        # x1 = self.Convv(x1)
        # x1 = x1 + x
        return self.final(x1)

class net2(nn.Module):
    def __init__(self,in_channels):
        super(net2, self).__init__()
        self.power = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=(3, 3), padding=1),

        )
        self.thetaphi = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 2, kernel_size=(3, 3), padding=1),

        )
        self.poweratio = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=(3, 3), padding=1),

        )
        self.losn = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=(3, 3), padding=1),

        )
        self.delay = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=(3, 3), padding=1),

        )
        self.sfmx = nn.Softmax(dim = 1)

    def forward(self,x):
        thephi = self.thetaphi(x)
        poweratio = self.poweratio(x)
        power = self.power(x)
        delay = self.delay(x)
        los = - torch.log(self.sfmx(self.losn(x)))

        return thephi,poweratio,power,delay,los
