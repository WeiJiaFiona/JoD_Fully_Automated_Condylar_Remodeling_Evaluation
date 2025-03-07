import os 
import json 
import torch 
import pandas as pd 
from glob import glob 
import torchio as tio 
import numpy as np 
from torch.utils import data as Data
import torch.nn.functional as F
from tqdm import tqdm 
from utils import collate_fn

class BoneSegmentationDataset(Data.Dataset):
    
    def __init__(self,
                 data_path, 
                 mode='train',
                 aug='resize',
    ):
        
        self.aug=aug
        self.mode = mode 
        self.prepare_data(data_path)


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        
        pid, img_path, lab_path = self.data[idx]['pid'], self.data[idx]['img'], self.data[idx]['lab']
        
        subject = tio.Subject(
            img = tio.ScalarImage(img_path),
            lab = tio.LabelMap(lab_path)
        )
        img = subject.img.data
        lab = subject.lab.data       
        if self.aug == 'resize':
        # 
            # resize
            img = F.interpolate(img.float().unsqueeze(0), size=(256, 256, 128), mode='trilinear').squeeze(0)
            lab = F.interpolate(lab.float().unsqueeze(0), size=(256, 256, 128), mode='trilinear').squeeze(0)

            return {
                'pid': self.data[idx]['pid'],
                'img': img,
                'lab': lab,

            }
            
        elif self.aug == 'crop' and self.mode in ['test']:
            # 512, 512, 128
            # crop (512, 512, 128)
            imgs = []
            i = 1
            while True:
                w, h, d = img.shape[1:]
                w_pad = 512 - w if 512 - w > 0 else 0
                h_pad = 512 - h if 512 - h > 0 else 0 
                d_pad = 128*i - d if 128*i - d > 0 else 0
                
                img = F.pad(img, (0,d_pad,0, h_pad, 0, w_pad),'constant', 0)
                lab = F.pad(lab, (0,d_pad,0, h_pad, 0, w_pad),'constant', 0)
            


                imgs.append(img[:, :512, :512, 128*(i-1):128*i].float())

                if 128 * i >= d:
                    break
                
                i += 1

            return {
                'pid': self.data[idx]['pid'],
                'ori': [w, h, d],
                'pad': [w_pad, h_pad, d_pad],
                'img': imgs,

            }                

        elif self.aug == 'crop' and self.mode in ['train', 'valid']:
            # crop (512, 512, 128)
            w, h, d = img.shape[1:]
            w_pad = 512 - w if 512 - w > 0 else 0
            h_pad = 512 - h if 512 - h > 0 else 0 
            d_pad = 128 - d if 128 - d > 0 else 0
            
            img = F.pad(img, (0,d_pad,0, h_pad, 0, w_pad),'constant', 0)
            lab = F.pad(lab, (0,d_pad,0, h_pad, 0, w_pad),'constant', 0)
            
            if w - 512 + w_pad > 0:
                w_s = np.random.randint(0, w-512 + w_pad)
            else:
                w_s = 0
            if h - 512 + h_pad > 0:
                h_s = np.random.randint(0, h-512 + h_pad)
            else:
                h_s = 0        
            if d - 128 + d_pad > 0:
                d_s = np.random.randint(0, d-128 + d_pad)
            else:
                d_s = 0     

            img = img[:, w_s:w_s + 512, h_s:h_s + 512, d_s:d_s + 128].float()
            lab = lab[:, w_s:w_s + 512, h_s:h_s + 512, d_s:d_s + 128].float()
        
            return {
                'pid': pid,
                'img': img,
                'lab': lab,

            }
            

    def check_state(self, table, pid):
        
        if pid not in table['Case'].values:
            return False
        else:
            return table[table['Case'] == pid]['State'].values == self.mode
    
    def get_loader(self, batch_size=1, shuffle=True):
        return Data.DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    
    def prepare_data(self, data_path):
        
        self.data = []
        
        # read table 
        state_table = pd.read_excel(data_path['state'])
        
        for path in tqdm(glob(os.path.join(data_path['img'], 'raw', '*'))):
            
            p_dir = os.path.basename(path)
            
            if not self.check_state(state_table, p_dir):
                continue 
            

            sample = {}
            
            sample['pid'] = p_dir
            sample['img'] = os.path.join(data_path['img'], 'raw', p_dir, 'image.nrrd')
            sample['lab'] = os.path.join(data_path['img'], 'raw', p_dir, 'label.nii.gz')
            
            self.data.append(sample)
            
