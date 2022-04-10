from __future__ import print_function
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.utils.data
# from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset,TensorDataset
from torchvision import datasets, transforms
import sys
import time
import mlt_dataprocess
import mlt_model
import mlt_train0
import mlt_train1
import resnet50
from thop import profile
from ptflops import get_model_complexity_info
import modelxp
import argparse
import mlt_train0
import mlt_train1
import mlt_train2

N_LOS_switch = False
state = 0
parser = argparse.ArgumentParser(description='Mltask prediction & classification')
parser.add_argument('--device',type=str,default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
parser.add_argument('--datapath',type=str,default='Data_all.npy')
parser.add_argument('--period',type=int,default=1)
parser.add_argument('--scale',type=int,default=2)
parser.add_argument('--test_ratio',type=float,default=0.2)
parser.add_argument('--seed',type=int,default=42)
parser.add_argument('--batch_size',type=int,default=1)
parser.add_argument('--step_size',type=int,default=30)
parser.add_argument('--aug_switch',type=bool,default=False)
parser.add_argument('--lr',type=float,default=1e-5)
parser.add_argument('--gamma',type=float,default=0.7)
parser.add_argument('--weight_decay',type=float,default=0)
parser.add_argument('--epochs_P1',type=int,default=100)
parser.add_argument('--epochs_P2',type=int,default=100)
# 修改关键参数4
keyinfo = 'Task10 MTL All-All-2-noite-nores-noaug'
characteristic_index = np.array([0,1,2,3,4,5,6])
target_index = np.array([1,2,3,4,5,6])

# 0 Height  1 keypara_angle_phi  2 keypara_angle_theta 3 keypara_k
# 4 keypara_p  5 keypara_t   6 keypara_LNLOS 1
if len([i for i in range(len(target_index)) if target_index[i] == 6 ]) > 0:
    N_LOS_switch = True
    state = 1
    if len(target_index) == 1:
        state = 2
parser.add_argument('--characteristic_index',type=np.ndarray,default=characteristic_index)
parser.add_argument('--target_index',type=np.ndarray,default=target_index)
parser.add_argument('--N_LOS_switch',type=bool,default=N_LOS_switch)
parser.add_argument('--state',type=int,default=state)
parser.add_argument('--logname',type=str,default=keyinfo +'.txt')
parser.add_argument('--jpgname',type=str,default=keyinfo +'.jpg')
parser.add_argument('--csvname',type=str,default=keyinfo +'.csv')
parser.add_argument('--argsname',type=str,default='args'+ keyinfo + '.txt')
parser.add_argument('--wight_lastname',type=str,default=keyinfo +'.pth')
args = parser.parse_args()

mlt_dataprocess.seed_everything(args)

# 第一次数据转化时需要
# vit_dataprocess.mat2npy()

# load & mask
channel_data_train, channel_data_test, mask_train, mask_test = mlt_dataprocess.Load_ChannelCharacteristicsData(args)
print('Training set shape:', channel_data_train.shape)
print('Test set shape:', channel_data_test.shape)

# interplate
Input_data_train = mlt_dataprocess.INterplate(channel_data_train, args.scale, 'bilinear')
Input_data_test = mlt_dataprocess.INterplate(channel_data_test, args.scale, 'bilinear')

# augumentation
if args.aug_switch == 1:
    channel_data_train = mlt_dataprocess.Dataugmentation(channel_data_train)
    # channel_data_test = mlt_dataprocess.Dataugmentation(channel_data_test)
    Input_data_train = mlt_dataprocess.Dataugmentation(Input_data_train)
    # Input_data_test = mlt_dataprocess.Dataugmentation(Input_data_test)
    mask_train = mlt_dataprocess.Dataugmentation(mask_train)
    # mask_test = mlt_dataprocess.Dataugmentation(mask_test)
print(channel_data_train.shape,channel_data_test.shape,Input_data_train.shape,Input_data_test.shape,
      mask_train.shape,mask_test.shape)

# wrap
test_loader = mlt_dataprocess.Dataset_Generator(channel_data_test, Input_data_test, mask_test,
                                                characteristic_index,target_index, args.batch_size, flag='False')
train_loader = mlt_dataprocess.Dataset_Generator(channel_data_train, Input_data_train, mask_train,
                                                 characteristic_index,target_index, args.batch_size, flag='True')
# test_loader = mlt_dataprocess.Dataset_Generator(channel_data_test, Input_data_test, mask_test,
#                                                 characteristic_index,target_index, args.batch_size, flag='False')
print('Length of Training set:',len(train_loader),'Length of Test set:',len(test_loader))

# model initialization
# model = modelxp.Discriminator().to(device)
# model = resnet50.ResNet50(len(characteristic_index)).to(device)
if args.state == 1:
    model = mlt_model.mltask(in_channels=len(characteristic_index), out_channels = 64 ).to(args.device)
else:
    model = mlt_model.mltask(in_channels=len(characteristic_index), out_channels = 64 ).to(args.device)

# model parameters and computation statistics
# print()
# dummy_input = torch.rand(1,7,200,200).to(device)
# flops,params = get_model_complexity_info(model,(7,200,200),as_strings=True,print_per_layer_stat=True)
# print('FLOPs:',flops,'params',params)
# print(summary(model,(7,200,200)))
# print()

# resume weights trained before
# model = vit_dataprocess.resume(model,os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result', 'temp.pth'))

# train
if state == 0:
    print('train0')
    mlt_train0.train(model,args,train_loader,test_loader)
elif state == 1:
    print('train1')
    mlt_train1.train(model,args,train_loader,test_loader)
elif state == 2:
    print('train2')
    mlt_train2.train(model,args,train_loader,test_loader)
else: raise KeyError

# save args
argsDict = args.__dict__
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result', args.argsname),'w') as f:
    f.writelines('--------------start---------------'+'\n')
    for eachArg,value in argsDict.items():
        f.writelines(eachArg+': '+str(value)+'\n')
    f.writelines('---------------end----------------')


