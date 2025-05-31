'''
To segment the mandible from coarse to fine
'''
import torch 
import torch.nn as nn
from model.bone_segmentation import BoneSegmentation, DiceLoss
from tqdm import tqdm
import numpy as np 
import pandas as pd 
import os
import argparse
import torch.nn.functional as F
import torchio as tio
from glob import glob 
from skimage.measure import label, regionprops, marching_cubes
import time
from pymeshlab import Mesh, MeshSet

class DiceLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target)
        z_sum = torch.sum(score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss
    
def voxel2mesh(img, spacing, smooth_iters=5, path=None):
    
    verts, faces, normals, values = marching_cubes(img, 0.5, spacing=spacing)
    mesh = Mesh(verts, faces)
    ms = MeshSet()
    ms.add_mesh(mesh)
    ms.invert_faces_orientation()
    ms.laplacian_smooth(stepsmoothnum=smooth_iters)

    if path is not None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ms.save_current_mesh(path)

def predict(path1, path2, img_path, out_path, tp, lab_path=None, normalize=(0, 2500)):

    # Stage 1: Initialize and load the first coarse VNet model for bone segmentation
    model = BoneSegmentation(1, 2)  
    model.load_state_dict(torch.load(path1)) 
    model.cuda()  
    model.eval()  
    
    start = time.time()
    img = tio.ScalarImage(img_path)  
    original_shape = img.data.shape[1:]
    target_spacing = np.array([0.48883, 0.48883, 1.])  
    target_size = tuple(int(x) for x in (np.array(img.data.shape[1:]) * np.array(img.spacing) / target_spacing).astype(int))

    # Interpolate image to unified space and normalize the intensity
    img_int = F.interpolate(img.data.unsqueeze(0).float(), size=tuple(target_size), mode='trilinear')
    img_int_numpy = img_int.squeeze().numpy()  
    img_int_numpy = np.clip(img_int_numpy, normalize[0], normalize[1])  
    img_int_norm = (img_int_numpy - img_int_numpy.min()) / (img_int_numpy.max() - img_int_numpy.min())  
    img_int_norm_tensor = F.interpolate(torch.from_numpy(img_int_norm).float().unsqueeze(0).unsqueeze(0), size=(256, 256, 128), mode='trilinear').float()
    with torch.no_grad():
        pred = model(img_int_norm_tensor.cuda())  
    pred_img = F.softmax(pred, dim=1)[:, 1].unsqueeze(0)  

    stage1_pred = F.interpolate(pred_img.float(), size=tuple(target_size), mode='trilinear').float()
    print('Prediction stage1:', time.time() - start)



    # Stage 2: Load the second VNet model for refining the prediction
    model = BoneSegmentation(1, 2)  
    model.load_state_dict(torch.load(path2))  
    model.cuda()  
    model.eval()  
    
    # Stage 2 Input: stage1_pred (label), Shape: (D, H, W) after stage 1 prediction
    start = time.time()
    stage1_pred_numpy = stage1_pred.cpu().squeeze().numpy()  
    stage1_pred_binary = np.where(stage1_pred_numpy > 0.3, 1, 0).astype(float)  
    label_numpy = label(stage1_pred_binary, connectivity=2)  
    regions = regionprops(label_numpy)  
    s_regions = sorted(regions, key=lambda x: x.area, reverse=True)  
    # Get the coordinates of the largest region (assumed to be the bone)
    points = np.array(np.where(label_numpy == s_regions[0].label))
    max_border = np.max(points, axis=1)
    min_border = np.min(points, axis=1)
    fixed_broaden = 20
    w = max_border[0] - min_border[0] + fixed_broaden
    h = max_border[1] - min_border[1] + fixed_broaden
    d = max_border[2] - min_border[2] + fixed_broaden
    
    w_pad = 32 - w % 32  
    h_pad = 32 - h % 32
    d_pad = 32 - d % 32
    
    ws = min_border[0] - fixed_broaden // 2 if min_border[0] - fixed_broaden // 2 > 0 else 0
    hs = min_border[1] - fixed_broaden // 2 if min_border[1] - fixed_broaden // 2 > 0 else 0
    ds = min_border[2] - fixed_broaden // 2 if min_border[2] - fixed_broaden // 2 > 0 else 0
    
    # Extract and pad the image region for stage 2
    stage2_img = torch.from_numpy(img_int_norm[ws:ws+w, hs:hs+h, ds:ds+d])
    stage2_img_pad = F.pad(stage2_img, (0, d_pad, 0, h_pad, 0, w_pad), 'constant', value=0).float().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        pred = model(stage2_img_pad.cuda())  
    pred_img = F.softmax(pred, dim=1)[:, 1]  
    # Remove padding and localize the predicted bone area
    pred_numpy = pred_img.cpu().squeeze().numpy()  
    pred_numpy_pad_remove = pred_numpy[:pred_numpy.shape[0]-w_pad, :pred_numpy.shape[1]-h_pad, :pred_numpy.shape[2]-d_pad]  # Remove padding

    # Create the final output mask
    stage2_out = np.zeros(tuple(target_size))  
    stage2_out[ws:ws+w, hs:hs+h, ds:ds+d] = pred_numpy_pad_remove  

    dice = 0
    if lab_path is not None:
        lab_img = tio.ScalarImage(lab_path).data.squeeze() 
        pre_img = F.interpolate(torch.from_numpy(stage2_out).float().unsqueeze(0).unsqueeze(0), size=img.data.shape[1:], mode='trilinear').float().squeeze()  # Interpolate to original space
        dice = 1 - DiceLoss()(pre_img, lab_img)  
    
    stage2_out = np.where(stage2_out > 0.3, 1, 0).astype(float)  
    stage2_out_label = label(stage2_out)  
    regions = sorted(regionprops(stage2_out_label), key=lambda x: x.area, reverse=True) 
    stage2_out_img = np.where(stage2_out_label == regions[0].label, 1, 0)  

    # Stage 2 Output: stage2_out_img (label), Shape: (D, H, W) after selecting the largest region

    # Interpolate the final output to original size
    out_img = F.interpolate(torch.from_numpy(stage2_out_img).float().unsqueeze(0).unsqueeze(0), size=img.data.shape[1:], mode='nearest').squeeze(0)  
    out_numpy = out_img.squeeze().numpy()  
    
    voxel2mesh(out_numpy, img.spacing, smooth_iters=5, path=os.path.join(out_path, f'{tp}_bone.off'))
    out_img = tio.ScalarImage(tensor=out_img, affine=img.affine)  
    os.makedirs(out_path, exist_ok=True)  
    out_img.save(os.path.join(out_path, f'{tp}_bone.nii.gz'))  

    print('Prediction stage2:', time.time() - start)

   
path1 = r'code\checkpoints\bone_segmentation\stage1.ckpt'
path2 = r'code\checkpoints\bone_segmentation\stage2.ckpt'


total_time = []
total_case = 0
tps = ['T1', 'T2']
dir_path = r'code\data'
IDs = os.listdir(dir_path)
for ID in IDs:
    for tp in tps:
        img_path = os.path.join(dir_path, ID, f'{tp}_img.nii.gz')
        img = tio.ScalarImage(img_path).data.numpy()
        normalize = (1024, 3524)
        if np.max(img) < 3500:
            normalize = (0, 2500)
        else:
            normalize = (1024, 3524)
        out_path = f'code/data/{ID}'
        predict(path1, path2, img_path, out_path, tp, lab_path=None, normalize=normalize)


    
    
    