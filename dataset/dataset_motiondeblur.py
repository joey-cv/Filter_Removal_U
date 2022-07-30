import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils import is_png_file, load_img, Augment_RGB_torch
import torch.nn.functional as F
import random
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import cv2
from natsort import natsorted
from glob import glob
augment   = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])
    
##################################################################################################
class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None):
        super(DataLoaderTrain, self).__init__()

        self.target_transform = target_transform
        self.train_files = rgb_dir
        self.files_name = sorted(os.listdir(self.train_files))

        self.noisy_filenames = []
        self.clean_filenames = []
        self.class_name = ['_1977.jpg', '_Amaro.jpg', '_Brannan.jpg', '_Clarendon.jpg', '_Gingham.jpg', '_He-Fe.jpg',
                           '_Hudson.jpg', '_Lo-Fi.jpg', '_Mayfair.jpg', '_Nashville.jpg', '_Perpetua.jpg',
                           '_Sutro.jpg', '_Toaster.jpg', '_Valencia.jpg', '_X-ProII.jpg']
        # self.class_name = ['_Amaro.jpg', '_Clarendon.jpg', '_Gingham.jpg', '_He-Fe.jpg',
        #  '_Hudson.jpg', '_Lo-Fi.jpg', '_Mayfair.jpg', '_Nashville.jpg', '_Perpetua.jpg',
        #  '_Valencia.jpg', '_X-ProII.jpg']
        for x in self.files_name:
            for cal_name in self.class_name:
                self.noisy_filenames.append(os.path.join(self.train_files, x, x + cal_name))
                self.clean_filenames.append(os.path.join(self.train_files, x, x + '_Original.jpg'))
        
        # clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        # noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))
        #
        # self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if is_png_file(x)]
        # self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x)       for x in noisy_files if is_png_file(x)]
        
        self.img_options=img_options

        self.tar_size = len(self.clean_filenames)  # get the size of target


    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
        
        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        #Crop Input and Target
        ps = self.img_options['patch_size']
        H = clean.shape[1]
        W = clean.shape[2]
        # r = np.random.randint(0, H - ps) if not H-ps else 0
        # c = np.random.randint(0, W - ps) if not H-ps else 0
        if H-ps==0:
            r=0
            c=0
        else:
            r = np.random.randint(0, H - ps)
            c = np.random.randint(0, W - ps)
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps]

        apply_trans = transforms_aug[random.getrandbits(3)]

        clean = getattr(augment, apply_trans)(clean)
        noisy = getattr(augment, apply_trans)(noisy)        

        return clean, noisy, noisy_filename, noisy_filename


##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform

        gt_dir = 'groundtruth'
        input_dir = 'input'
        
        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))


        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if is_png_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files if is_png_file(x)]
        

        self.tar_size = len(self.clean_filenames)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        

        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
                
        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)

        return clean, noisy, clean_filename, noisy_filename

class DataLoaderVal_deblur(Dataset):
    def __init__(self, rgb_dir, img_options=None, rgb_dir2=None):
        super(DataLoaderVal_deblur, self).__init__()

        # inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        # tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'groundtruth')))
        #
        # self.inp_filenames = [os.path.join(rgb_dir, 'input', x)  for x in inp_files if is_png_file(x)]
        # self.tar_filenames = [os.path.join(rgb_dir, 'groundtruth', x) for x in tar_files if is_png_file(x)]

        self.train_files = rgb_dir
        self.files_name = sorted(os.listdir(self.train_files))

        self.inp_filenames = []
        self.tar_filenames = []
        # self.class_name = ['_1977.jpg', '_Amaro.jpg', '_Brannan.jpg', '_Clarendon.jpg', '_Gingham.jpg', '_He-Fe.jpg',
        #                    '_Hudson.jpg', '_Lo-Fi.jpg', '_Mayfair.jpg', '_Nashville.jpg', '_Perpetua.jpg',
        #                    '_Sutro.jpg', '_Toaster.jpg', '_Valencia.jpg', '_X-ProII.jpg']
        self.class_name = ['_Amaro.jpg', '_Clarendon.jpg', '_Gingham.jpg', '_He-Fe.jpg',
         '_Hudson.jpg', '_Lo-Fi.jpg', '_Mayfair.jpg', '_Nashville.jpg', '_Perpetua.jpg',
         '_Valencia.jpg', '_X-ProII.jpg']
        for x in self.files_name:
            for cal_name in self.class_name:
                self.inp_filenames.append(os.path.join(self.train_files, x, x + cal_name))
                self.tar_filenames.append(os.path.join(self.train_files, x, x + '_Original.jpg'))

        self.img_options = img_options
        self.tar_size       = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options['patch_size'] if img_options is not None else None

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        index_ = index % self.tar_size
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path)
        tar_img = Image.open(tar_path)

        # Validate on center crop
        # if self.ps is not None:
        #     inp_img = TF.center_crop(inp_img, (ps,ps))
        #     tar_img = TF.center_crop(tar_img, (ps,ps))

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        filename = os.path.splitext(os.path.split(inp_path)[-1])[0]

        return tar_img, inp_img, filename
        
##################################################################################################

class DataLoaderTest(Dataset):
    def __init__(self, inp_dir, img_options):
        super(DataLoaderTest, self).__init__()

        # self.inp_filenames = [os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)]
        self.val_files = inp_dir
        self.aug_num = 0
        self.inp_files = sorted(os.listdir(self.val_files))
        # self.class_name = ['_1977.jpg', '_Amaro.jpg', '_Brannan.jpg', '_Clarendon.jpg', '_Gingham.jpg', '_He-Fe.jpg',
        #                    '_Hudson.jpg', '_Lo-Fi.jpg', '_Mayfair.jpg', '_Nashville.jpg', '_Perpetua.jpg',
        #                    '_Sutro.jpg', '_Toaster.jpg', '_Valencia.jpg', '_X-ProII.jpg']
        self.class_name = ['_Amaro.jpg', '_Clarendon.jpg', '_Gingham.jpg', '_He-Fe.jpg',
                           '_Hudson.jpg', '_Lo-Fi.jpg', '_Mayfair.jpg', '_Nashville.jpg', '_Perpetua.jpg',
                           '_Valencia.jpg', '_X-ProII.jpg']
        self.inp_filenames = []
        for x in self.inp_files:
            for cal_name in self.class_name:
                self.inp_filenames.append(os.path.join(self.val_files, x, x+cal_name))

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options
        self.img_normalize=transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):

        path_inp = self.inp_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
        inp = Image.open(path_inp)
        inp = TF.to_tensor(inp)

        apply_trans = transforms_aug[self.aug_num]
        inp = getattr(augment, apply_trans)(inp)

        # src_img = cv2.imread(path_inp)
        # (b, g, r) = cv2.split(src_img)
        #
        #
        # bH = cv2.equalizeHist(b)
        # gH = cv2.equalizeHist(g)
        # rH = cv2.equalizeHist(r)
        # out_img = cv2.merge([bH, gH, rH])
        # inp = TF.to_tensor(out_img)
        # inp = self.img_normalize(inp)
        return inp, filename


def get_training_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, img_options, None)


def get_validation_deblur_data(rgb_dir, img_options=None):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal_deblur(rgb_dir, img_options, None)

def get_test_data(rgb_dir, img_options=None):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, img_options)