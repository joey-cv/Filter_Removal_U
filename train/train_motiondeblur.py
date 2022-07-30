import os
import sys

# add dir
dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'../dataset/'))
sys.path.append(os.path.join(dir_name,'..'))
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

######### Logs dir ###########
log_dir = os.path.join(opt.save_dir,'motiondeblur',opt.dataset, opt.arch+opt.env)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
logname = os.path.join(log_dir, datetime.datetime.now().isoformat()+'.txt') 
print("Now time is : ",datetime.datetime.now().isoformat())
result_dir = os.path.join(log_dir, 'results')
model_dir  = os.path.join(log_dir, 'models')
utils.mkdir(result_dir)
utils.mkdir(model_dir)

# ######### Set Seeds ###########
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

######### Model ###########
model_restoration = utils.get_arch(opt)

with open(logname,'a') as f:
    f.write(str(opt)+'\n')
    f.write(str(model_restoration)+'\n')

######### Optimizer ###########
start_epoch = 1
if opt.optimizer.lower() == 'adam':
    optimizer = optim.Adam(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
elif opt.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(model_restoration.parameters(), lr=opt.lr_initial, betas=(0.9, 0.999),eps=1e-8, weight_decay=opt.weight_decay)
else:
    raise Exception("Error optimizer...")


######### DataParallel ########### 
model_restoration = torch.nn.DataParallel (model_restoration)
print("cuda.is_available:{}".format(torch.cuda.is_available()))
model_restoration.cuda() 
     

######### Scheduler ###########
if opt.warmup:
    print("Using warmup and cosine strategy!")
    warmup_epochs = opt.warmup_epochs
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-warmup_epochs, eta_min=1e-6)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()
else:
    step = 0
    print("Using StepLR,step={}!".format(step))
    scheduler = StepLR(optimizer, step_size=step, gamma=0.5)
    scheduler.step()

######### Resume ########### 
if opt.resume:
    path_chk_rest = opt.pretrain_weights 
    print("Resume from "+path_chk_rest)
    utils.load_checkpoint(model_restoration,path_chk_rest) 
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1 
    lr = utils.load_optim(optimizer, path_chk_rest) 

    # for p in optimizer.param_groups: p['lr'] = lr 
    # warmup = False 
    # new_lr = lr 
    # print('------------------------------------------------------------------------------') 
    # print("==> Resuming Training with learning rate:",new_lr) 
    # print('------------------------------------------------------------------------------') 
    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')

    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.nepoch-start_epoch+1, eta_min=1e-6) 

######### Loss ###########
criterion = CharbonnierLoss().cuda()

######### DataLoader ###########
print('===> Loading datasets')
img_options_train = {'patch_size':opt.train_ps}
train_dataset = get_training_data(opt.train_dir, img_options_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, shuffle=True, 
        num_workers=opt.train_workers, pin_memory=False, drop_last=False)

img_options_val = {'patch_size':opt.val_ps}
val_dataset = get_validation_deblur_data(opt.val_dir, img_options_val)

val_loader = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, shuffle=False, 
        num_workers=opt.eval_workers, pin_memory=False, drop_last=False)

len_trainset = train_dataset.__len__()
len_valset = val_dataset.__len__()
print("Sizeof training set: ", len_trainset,", sizeof validation set: ", len_valset)
######### validation ###########
# with torch.no_grad():
#     model_restoration.eval()
#     psnr_dataset = []
#     psnr_model_init = []
#     for ii, data_val in enumerate((val_loader), 0):
#         target = data_val[0].cuda()
#         input_ = data_val[1].cuda()
#         with torch.cuda.amp.autocast():
#             restored = model_restoration(input_)
#             restored = torch.clamp(restored,0,1)
#         psnr_dataset.append(utils.batch_PSNR(input_, target, False).item())
#         psnr_model_init.append(utils.batch_PSNR(restored, target, False).item())
#     psnr_dataset = sum(psnr_dataset)/len_valset
#     psnr_model_init = sum(psnr_model_init)/len_valset
#     print('Input & GT (PSNR) -->%.4f dB'%(psnr_dataset), ', Model_init & GT (PSNR) -->%.4f dB'%(psnr_model_init))

######### train ###########
print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.nepoch))
best_psnr = 0
best_epoch = 0
best_iter = 0
# eval_now = len(train_loader)//4
eval_now = 1
print("\nEvaluation after every {} epoch !!!\n".format(eval_now))

loss_scaler = NativeScaler()
torch.cuda.empty_cache()
for epoch in range(start_epoch, opt.nepoch + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1
    psnr_val_rgb = []
    filter_scores = [0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0,
                     0, 0, 0, 0, 0]
    for i, data in enumerate(tqdm(train_loader), 0):
        # zero_grad
        optimizer.zero_grad()

        target = data[0].cuda()
        input_ = data[1].cuda()
        filenames = data[2]

        with torch.cuda.amp.autocast():
            restored = model_restoration(input_)
            loss = criterion(restored, target)
        loss_scaler(
                loss, optimizer,parameters=model_restoration.parameters())
        epoch_loss +=loss.item()

        for res, tar, fname in zip(restored, target, filenames):
            # psnr_scores = utils.batch_PSNR(res, tar, False)
            psnr_scores = utils.myPSNR(res, tar)
            psnr_val_rgb.append(psnr_scores)
            fname_ = fname.split('_')[1]
            fname_ = fname_.split('.')[0]
            if fname_ == '1977':
                filter_scores[0] = filter_scores[0] + psnr_scores
            elif fname_ == 'Amaro':
                filter_scores[1] = filter_scores[1] + psnr_scores
            elif fname_ == 'Brannan':
                filter_scores[2] = filter_scores[2] + psnr_scores
            elif fname_ == 'Clarendon':
                filter_scores[3] = filter_scores[3] + psnr_scores
            elif fname_ == 'Gingham':
                filter_scores[4] = filter_scores[4] + psnr_scores
            elif fname_ == 'He-Fe':
                filter_scores[5] = filter_scores[5] + psnr_scores
            elif fname_ == 'Hudson':
                filter_scores[6] = filter_scores[6] + psnr_scores
            elif fname_ == 'Lo-Fi':
                filter_scores[7] = filter_scores[7] + psnr_scores
            elif fname_ == 'Mayfair':
                filter_scores[8] = filter_scores[8] + psnr_scores
            elif fname_ == 'Nashville':
                filter_scores[9] = filter_scores[9] + psnr_scores
            elif fname_ == 'Perpetua':
                filter_scores[10] = filter_scores[10] + psnr_scores
            elif fname_ == 'Sutro':
                filter_scores[11] = filter_scores[11] + psnr_scores
            elif fname_ == 'Toaster':
                filter_scores[12] = filter_scores[12] + psnr_scores
            elif fname_ == 'Valencia':
                filter_scores[13] = filter_scores[13] + psnr_scores
            elif fname_ == 'X-ProII':
                filter_scores[14] = filter_scores[14] + psnr_scores
        # psnr_val_rgb.append(utils.batch_PSNR(restored, target, False).item())
    filter_scores = [filter_score / 500 for filter_score in filter_scores]
    psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
    print(
        "[epoch %d 1977_PSNR: %.4f Amaro_PSNR: %.4f Brannan_PSNR: %.4f Clarendon_PSNR: %.4f] Gingham_PSNR: %.4f]"
        % (epoch, filter_scores[0], filter_scores[1], filter_scores[2], filter_scores[3], filter_scores[4]))
    print(
        "[epoch %d He-Fe_PSNR: %.4f Hudson_PSNR: %.4f Lo-Fi_PSNR: %.4f Mayfair_PSNR: %.4f] Nashville_PSNR: %.4f]"
        % (epoch, filter_scores[5], filter_scores[6], filter_scores[7], filter_scores[8], filter_scores[9]))
    print(
        "[epoch %d Perpetua_PSNR: %.4f Sutro_PSNR: %.4f Toaster_PSNR: %.4f Valencia_PSNR: %.4f] X-ProII_PSNR: %.4f]"
        % (
            epoch, filter_scores[10], filter_scores[11], filter_scores[12], filter_scores[13], filter_scores[14]))
    print("[Ep %d \t PSNR SIDD: %.4f\t]" % (
        epoch, psnr_val_rgb))
        # #### Evaluation ####
        # if (i+1)%eval_now==0 and i>0:
        #     with torch.no_grad():
        #         model_restoration.eval()
        #         psnr_val_rgb = []
        #         for ii, data_val in enumerate((val_loader), 0):
        #             target = data_val[0].cuda()
        #             input_ = data_val[1].cuda()
        #             filenames = data_val[2]
        #             with torch.cuda.amp.autocast():
        #                 restored = model_restoration(input_)
        #             restored = torch.clamp(restored,0,1)
        #             psnr_val_rgb.append(utils.batch_PSNR(restored, target, False).item())
        #
        #         psnr_val_rgb = sum(psnr_val_rgb)/len_valset
        #
        #         if psnr_val_rgb > best_psnr:
        #             best_psnr = psnr_val_rgb
        #             best_epoch = epoch
        #             best_iter = i
        #             torch.save({'epoch': epoch,
        #                         'state_dict': model_restoration.state_dict(),
        #                         'optimizer' : optimizer.state_dict()
        #                         }, os.path.join(model_dir,"model_best.pth"))
        #
        #         print("[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " % (epoch, i, psnr_val_rgb,best_epoch,best_iter,best_psnr))
        #         with open(logname,'a') as f:
        #             f.write("[Ep %d it %d\t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d best_it_SIDD %d Best_PSNR_SIDD %.4f] " \
        #                 % (epoch, i, psnr_val_rgb,best_epoch,best_iter,best_psnr)+'\n')
        #         model_restoration.train()
        #         torch.cuda.empty_cache()
    scheduler.step()

    #### Evaluation ####
    if (epoch) % eval_now == 0 and epoch > 1000:
        with torch.no_grad():
            model_restoration.eval()
            psnr_val_rgb = []
            # 1977 Amaro Brannan Clarendon Gingham
            # He-Fe Hudson Lo-Fi Mayfair Nashville
            # Perpetua Sutro Toaster Valencia X-ProII
            filter_scores = [0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0,
                             0, 0, 0, 0, 0]
            for ii, data_val in enumerate((val_loader), 0):
                target = data_val[0].cuda()
                input_ = data_val[1].cuda()
                filenames = data_val[2]
                with torch.cuda.amp.autocast():
                    restored = model_restoration(input_)
                restored = torch.clamp(restored, 0, 1)

                for res, tar, fname in zip(restored, target, filenames):
                    # psnr_scores = utils.batch_PSNR(res, tar, False)
                    psnr_scores = utils.myPSNR(res, tar)
                    psnr_val_rgb.append(psnr_scores)
                    fname_ = fname.split('_')[1]
                    if fname_ == '1977':
                        filter_scores[0] = filter_scores[0] + psnr_scores
                    elif fname_ == 'Amaro':
                        filter_scores[1] = filter_scores[1] + psnr_scores
                    elif fname_ == 'Brannan':
                        filter_scores[2] = filter_scores[2] + psnr_scores
                    elif fname_ == 'Clarendon':
                        filter_scores[3] = filter_scores[3] + psnr_scores
                    elif fname_ == 'Gingham':
                        filter_scores[4] = filter_scores[4] + psnr_scores
                    elif fname_ == 'He-Fe':
                        filter_scores[5] = filter_scores[5] + psnr_scores
                    elif fname_ == 'Hudson':
                        filter_scores[6] = filter_scores[6] + psnr_scores
                    elif fname_ == 'Lo-Fi':
                        filter_scores[7] = filter_scores[7] + psnr_scores
                    elif fname_ == 'Mayfair':
                        filter_scores[8] = filter_scores[8] + psnr_scores
                    elif fname_ == 'Nashville':
                        filter_scores[9] = filter_scores[9] + psnr_scores
                    elif fname_ == 'Perpetua':
                        filter_scores[10] = filter_scores[10] + psnr_scores
                    elif fname_ == 'Sutro':
                        filter_scores[11] = filter_scores[11] + psnr_scores
                    elif fname_ == 'Toaster':
                        filter_scores[12] = filter_scores[12] + psnr_scores
                    elif fname_ == 'Valencia':
                        filter_scores[13] = filter_scores[13] + psnr_scores
                    elif fname_ == 'X-ProII':
                        filter_scores[14] = filter_scores[14] + psnr_scores
                # psnr_val_rgb.append(utils.batch_PSNR(restored, target, False).item())
            filter_scores = [filter_score / 44 for filter_score in filter_scores]
            psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()

            if psnr_val_rgb > best_psnr:
                best_psnr = psnr_val_rgb
                best_epoch = epoch
                torch.save({'epoch': epoch,
                            'state_dict': model_restoration.state_dict(),
                            'optimizer': optimizer.state_dict()
                            }, os.path.join(model_dir, "model_best.pth"))

            print(
                "[epoch %d 1977_PSNR: %.4f Amaro_PSNR: %.4f Brannan_PSNR: %.4f Clarendon_PSNR: %.4f] Gingham_PSNR: %.4f]"
                % (epoch, filter_scores[0], filter_scores[1], filter_scores[2], filter_scores[3], filter_scores[4]))
            print(
                "[epoch %d He-Fe_PSNR: %.4f Hudson_PSNR: %.4f Lo-Fi_PSNR: %.4f Mayfair_PSNR: %.4f] Nashville_PSNR: %.4f]"
                % (epoch, filter_scores[5], filter_scores[6], filter_scores[7], filter_scores[8], filter_scores[9]))
            print(
                "[epoch %d Perpetua_PSNR: %.4f Sutro_PSNR: %.4f Toaster_PSNR: %.4f Valencia_PSNR: %.4f] X-ProII_PSNR: %.4f]"
                % (
                epoch, filter_scores[10], filter_scores[11], filter_scores[12], filter_scores[13], filter_scores[14]))
            print("[Ep %d \t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d Best_PSNR_SIDD %.4f] " % (
            epoch, psnr_val_rgb, best_epoch, best_psnr))
            with open(logname, 'a') as f:
                f.write("[Ep %d \t PSNR SIDD: %.4f\t] ----  [best_Ep_SIDD %d Best_PSNR_SIDD %.4f] " \
                        % (epoch, psnr_val_rgb, best_epoch, best_psnr) + '\n')
            model_restoration.train()
            torch.cuda.empty_cache()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")
    with open(logname,'a') as f:
        f.write("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time()-epoch_start_time,epoch_loss, scheduler.get_lr()[0])+'\n')

    torch.save({'epoch': epoch, 
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,"model_latest.pth"))   

    if epoch%opt.checkpoint == 0:
        torch.save({'epoch': epoch, 
                    'state_dict': model_restoration.state_dict(),
                    'optimizer' : optimizer.state_dict()
                    }, os.path.join(model_dir,"model_epoch_{}.pth".format(epoch))) 
print("Now time is : ",datetime.datetime.now().isoformat())
