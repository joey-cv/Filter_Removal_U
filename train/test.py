import os
import sys
from skimage import img_as_ubyte
import shutil
# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name, '../dataset/'))
sys.path.append(os.path.join(dir_name, '..'))
print(dir_name)

import argparse
import options

######### parser ###########
opt = options.Options().init(argparse.ArgumentParser(description='Image motion deblurring')).parse_args()
print(opt)

import utils
from dataset.dataset_motiondeblur import *

######### Set GPUs ###########
print("cuda.is_available:{}".format(torch.cuda.is_available()))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
import torch

torch.backends.cudnn.benchmark = True

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from natsort import natsorted
import glob
import random
import time
import numpy as np
from einops import rearrange, repeat
import datetime
from pdb import set_trace as stx

from losses import CharbonnierLoss

from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from timm.utils import NativeScaler
from model import ModelOver



######### Logs dir ###########
log_dir = os.path.join(opt.save_dir, 'motiondeblur', opt.dataset, opt.arch + opt.env)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logname = os.path.join(log_dir, datetime.datetime.now().isoformat() + '.txt')
print("Now time is : ", datetime.datetime.now().isoformat())
result_dir = opt.result_dir
result_dir1 = opt.result_dir1
result_dir2 = opt.result_dir2
result_dir3 = opt.result_dir3
result_dir4 = opt.result_dir4
result_dir5 = opt.result_dir5
result_dir6 = opt.result_dir6
result_dir7 = opt.result_dir7
utils.mkdir(result_dir)
modelover = ModelOver(result_dir, result_dir1, result_dir2, result_dir3, result_dir4,
                      result_dir5, result_dir6, result_dir7, opt.fusion_file, opt.end_file)

# ######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

######### Model ###########
model_restoration = utils.get_arch(opt)

with open(logname, 'a') as f:
    f.write(str(opt) + '\n')
    f.write(str(model_restoration) + '\n')

######### Optimizer ###########
start_epoch = 1
if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999), eps=1e-8,
                           weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'adamw':
    optimizer = optim.AdamW(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=opt.weight_decay)
else:
    raise Exception("Error optimizer...")

######### DataParallel ###########
model_restoration = torch.nn.DataParallel(model_restoration)
print("cuda.is_available:{}".format(torch.cuda.is_available()))
model_restoration.cuda()


######### Resume ###########
if opt.resume:
    path_chk_rest = opt.pretrain_weights
    print("Resume from " + path_chk_rest)
    utils.load_checkpoint(model_restoration, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    lr = utils.load_optim(optimizer, path_chk_rest)

######### DataLoader ###########
print('===> Loading datasets')
img_options_train = {'patch_size': opt.train_ps}
test_dataset = get_test_data(opt.test_dir, img_options_train)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False,
                          num_workers=4, pin_memory=True, drop_last=False)

change_file_dirs =[90,91,92,93,94,95,96,97,98,99]

def just_data(file_root, change_file_dirs):
    if os.path.exists(file_root):
        for change_file_dir in change_file_dirs:
            change_file_name = str(change_file_dir) + '_Hudson.png'
            if change_file_dir != 90:
                out_file_dir = change_file_dir - 1
            else:
                out_file_dir = 99
            src_file_path = os.path.join(file_root, str(change_file_dir), change_file_name)
            out_file_path = os.path.join(file_root, str(out_file_dir), change_file_name)
            shutil.move(src_file_path, out_file_path)

        for change_file_dir in change_file_dirs:
            if change_file_dir != 99:
                out_file_name = change_file_dir + 1
            else:
                out_file_name = 90
            src_file_name = str(out_file_name) + '_Hudson.png'
            out_file_name = str(change_file_dir) + '_Hudson.png'
            src_file_path = os.path.join(file_root, str(change_file_dir), src_file_name)
            out_file_path = os.path.join(file_root, str(change_file_dir), out_file_name)
            shutil.move(src_file_path, out_file_path)
    else:
        print('no exists src')

######### test ###########
with torch.no_grad():
    model_restoration.eval()
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        input_ = data_test[0].cuda()
        filenames = data_test[1]

        restored = model_restoration(input_)
        restored = torch.clamp(restored, 0, 1)

        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        for batch in range(len(restored)):
            restored_img = img_as_ubyte(restored[batch])
            keep_dir = os.path.join(result_dir, filenames[0].split('_')[0])
            # print(keep_dir)
            if not os.path.exists(keep_dir):
                os.makedirs(keep_dir)
            utils.save_img((os.path.join(keep_dir, filenames[batch]+'.png')), restored_img)

    just_data(result_dir, change_file_dirs)
    test_loader.dataset.aug_num = test_loader.dataset.aug_num+1
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        input_ = data_test[0].cuda()
        filenames = data_test[1]

        restored = model_restoration(input_)
        restored = torch.clamp(restored, 0, 1)

        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        for batch in range(len(restored)):
            restored_img = img_as_ubyte(restored[batch])
            keep_dir = os.path.join(result_dir1, filenames[0].split('_')[0])
            # print(keep_dir)
            if not os.path.exists(keep_dir):
                os.makedirs(keep_dir)
            utils.save_img((os.path.join(keep_dir, filenames[batch]+'.png')), restored_img)

    just_data(result_dir1, change_file_dirs)
    test_loader.dataset.aug_num = test_loader.dataset.aug_num+1
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        input_ = data_test[0].cuda()
        filenames = data_test[1]

        restored = model_restoration(input_)
        restored = torch.clamp(restored, 0, 1)

        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        for batch in range(len(restored)):
            restored_img = img_as_ubyte(restored[batch])
            keep_dir = os.path.join(result_dir2, filenames[0].split('_')[0])
            # print(keep_dir)
            if not os.path.exists(keep_dir):
                os.makedirs(keep_dir)

            utils.save_img((os.path.join(keep_dir, filenames[batch]+'.png')), restored_img)

    just_data(result_dir2, change_file_dirs)
    test_loader.dataset.aug_num = test_loader.dataset.aug_num+1
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        input_ = data_test[0].cuda()
        filenames = data_test[1]

        restored = model_restoration(input_)
        restored = torch.clamp(restored, 0, 1)

        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        for batch in range(len(restored)):
            restored_img = img_as_ubyte(restored[batch])
            keep_dir = os.path.join(result_dir3, filenames[0].split('_')[0])
            # print(keep_dir)
            if not os.path.exists(keep_dir):
                os.makedirs(keep_dir)

            utils.save_img((os.path.join(keep_dir, filenames[batch]+'.png')), restored_img)

    just_data(result_dir3, change_file_dirs)
    test_loader.dataset.aug_num = test_loader.dataset.aug_num+1
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        input_ = data_test[0].cuda()
        filenames = data_test[1]

        restored = model_restoration(input_)
        restored = torch.clamp(restored, 0, 1)

        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        for batch in range(len(restored)):
            restored_img = img_as_ubyte(restored[batch])
            keep_dir = os.path.join(result_dir4, filenames[0].split('_')[0])
            # print(keep_dir)
            if not os.path.exists(keep_dir):
                os.makedirs(keep_dir)

            utils.save_img((os.path.join(keep_dir, filenames[batch]+'.png')), restored_img)

    just_data(result_dir4, change_file_dirs)
    test_loader.dataset.aug_num = test_loader.dataset.aug_num+1
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        input_ = data_test[0].cuda()
        filenames = data_test[1]

        restored = model_restoration(input_)
        restored = torch.clamp(restored, 0, 1)

        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        for batch in range(len(restored)):
            restored_img = img_as_ubyte(restored[batch])
            keep_dir = os.path.join(result_dir5, filenames[0].split('_')[0])
            # print(keep_dir)
            if not os.path.exists(keep_dir):
                os.makedirs(keep_dir)

            utils.save_img((os.path.join(keep_dir, filenames[batch]+'.png')), restored_img)

    just_data(result_dir5, change_file_dirs)
    test_loader.dataset.aug_num = test_loader.dataset.aug_num+1
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        input_ = data_test[0].cuda()
        filenames = data_test[1]

        restored = model_restoration(input_)
        restored = torch.clamp(restored, 0, 1)

        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        for batch in range(len(restored)):
            restored_img = img_as_ubyte(restored[batch])
            keep_dir = os.path.join(result_dir6, filenames[0].split('_')[0])
            # print(keep_dir)
            if not os.path.exists(keep_dir):
                os.makedirs(keep_dir)

            utils.save_img((os.path.join(keep_dir, filenames[batch]+'.png')), restored_img)

    just_data(result_dir6, change_file_dirs)
    test_loader.dataset.aug_num = test_loader.dataset.aug_num+1
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        input_ = data_test[0].cuda()
        filenames = data_test[1]

        restored = model_restoration(input_)
        restored = torch.clamp(restored, 0, 1)

        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        for batch in range(len(restored)):
            restored_img = img_as_ubyte(restored[batch])
            keep_dir = os.path.join(result_dir7, filenames[0].split('_')[0])
            # print(keep_dir)
            if not os.path.exists(keep_dir):
                os.makedirs(keep_dir)

            utils.save_img((os.path.join(keep_dir, filenames[batch]+'.png')), restored_img)

    just_data(result_dir7, change_file_dirs)
    modelover.finish_over()




