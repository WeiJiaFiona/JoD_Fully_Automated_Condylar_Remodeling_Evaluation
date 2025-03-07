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

    # Stage 1: Initialize and load the first VNet model for bone segmentation
    model = BoneSegmentation(1, 2)  # Binary segmentation (1 input channel, 2 classes)
    model.load_state_dict(torch.load(path1))  # Load the model weights for stage 1
    model.cuda()  # Move the model to GPU
    model.eval()  # Set the model to evaluation mode
    
    start = time.time()
    img = tio.ScalarImage(img_path)  # Load the input image using torchio (3D image)
    
    # interpolate to unified space and normalize
        
    # Stage 1 Input: img (image), Shape: (C, D, H, W) where C is 1 (grayscale), 
    # D, H, W are the depth, height, and width of the 3D image
    # Normalize the image and rescale it to a uniform space (target_spacing)
    original_shape = img.data.shape[1:]
    target_spacing = np.array([0.48883, 0.48883, 1.])  # Set the target spacing (in mm)
    target_size = tuple(int(x) for x in (np.array(img.data.shape[1:]) * np.array(img.spacing) / target_spacing).astype(int))
    # print(f"Stage 1 Input: {img.data.shape}, Target Size: {target_size}")

    # Interpolate image to unified space and normalize the intensity
    img_int = F.interpolate(img.data.unsqueeze(0).float(), size=tuple(target_size), mode='trilinear')
    img_int_numpy = img_int.squeeze().numpy()  # Convert to numpy for further processing
    img_int_numpy = np.clip(img_int_numpy, normalize[0], normalize[1])  # Intensity normalization
    img_int_norm = (img_int_numpy - img_int_numpy.min()) / (img_int_numpy.max() - img_int_numpy.min())  # Normalize to [0, 1]

    # Stage 1 Output: img_int_norm (image), Shape: (D, H, W) after interpolation
    img_int_norm_tensor = F.interpolate(torch.from_numpy(img_int_norm).float().unsqueeze(0).unsqueeze(0), size=(256, 256, 128), mode='trilinear').float()

    
    
    # Perform prediction using VNet model
    with torch.no_grad():
        pred = model(img_int_norm_tensor.cuda())  # Pass the image through the model
    pred_img = F.softmax(pred, dim=1)[:, 1].unsqueeze(0)  # Get the prediction of class 1 (bone)

    # Stage 1 Output: pred_img (image), Shape: (1, D, H, W) after VNet prediction
    # Interpolate the prediction to the original image space (target_size)
    stage1_pred = F.interpolate(pred_img.float(), size=tuple(target_size), mode='trilinear').float()
    print('Prediction stage1:', time.time() - start)
    # print('Stage 1 inferred output shape', stage1_pred.shape)  # Stage 1 output shape


    # Stage 2: Load the second VNet model for refining the prediction
    model = BoneSegmentation(1, 2)  # Binary segmentation (1 input channel, 2 classes)
    model.load_state_dict(torch.load(path2))  # Load the model weights for stage 2
    model.cuda()  # Move the model to GPU
    model.eval()  # Set the model to evaluation mode
    
    # Stage 2 Input: stage1_pred (label), Shape: (D, H, W) after stage 1 prediction
    start = time.time()
    # Convert the stage 1 prediction into a binary mask
    stage1_pred_numpy = stage1_pred.cpu().squeeze().numpy()  # Convert to numpy array
    stage1_pred_binary = np.where(stage1_pred_numpy > 0.3, 1, 0).astype(float)  # Thresholding for binary segmentation
    
    label_numpy = label(stage1_pred_binary, connectivity=2)  # Label connected components in the binary mask
    regions = regionprops(label_numpy)  # Get region properties
    s_regions = sorted(regions, key=lambda x: x.area, reverse=True)  # Sort regions by area (largest to smallest)
    
    # Get the coordinates of the largest region (assumed to be the bone)
    points = np.array(np.where(label_numpy == s_regions[0].label))
    
    # Calculate bounding box and add padding for stage 2
    max_border = np.max(points, axis=1)
    min_border = np.min(points, axis=1)
    fixed_broaden = 20
    w = max_border[0] - min_border[0] + fixed_broaden
    h = max_border[1] - min_border[1] + fixed_broaden
    d = max_border[2] - min_border[2] + fixed_broaden
    
    # Stage 2 Input: label_numpy (label), Shape: (D, H, W) after labeling the largest bone region
    
    w_pad = 32 - w % 32  # Pad to make dimensions divisible by 32
    h_pad = 32 - h % 32
    d_pad = 32 - d % 32
    
    ws = min_border[0] - fixed_broaden // 2 if min_border[0] - fixed_broaden // 2 > 0 else 0
    hs = min_border[1] - fixed_broaden // 2 if min_border[1] - fixed_broaden // 2 > 0 else 0
    ds = min_border[2] - fixed_broaden // 2 if min_border[2] - fixed_broaden // 2 > 0 else 0
    
    # Extract and pad the image region for stage 2
    stage2_img = torch.from_numpy(img_int_norm[ws:ws+w, hs:hs+h, ds:ds+d])
    stage2_img_pad = F.pad(stage2_img, (0, d_pad, 0, h_pad, 0, w_pad), 'constant', value=0).float().unsqueeze(0).unsqueeze(0)

    # Perform prediction using the second VNet model
    with torch.no_grad():
        pred = model(stage2_img_pad.cuda())  # Pass the image through the model
    pred_img = F.softmax(pred, dim=1)[:, 1]  # Get the prediction of class 1 (bone)

    # Stage 2 Output: pred_img (image), Shape: (1, D, H, W) after VNet prediction

    # Remove padding and localize the predicted bone area
    pred_numpy = pred_img.cpu().squeeze().numpy()  # Convert prediction to numpy
    pred_numpy_pad_remove = pred_numpy[:pred_numpy.shape[0]-w_pad, :pred_numpy.shape[1]-h_pad, :pred_numpy.shape[2]-d_pad]  # Remove padding

    # Stage 2 Output: pred_numpy_pad_remove (image), Shape: (D, H, W) after removing padding

    # Create the final output mask
    stage2_out = np.zeros(tuple(target_size))  # Initialize a zero mask
    stage2_out[ws:ws+w, hs:hs+h, ds:ds+d] = pred_numpy_pad_remove  # Place the predicted mask in the original size

    # Stage 2 Output: stage2_out (label), Shape: (D, H, W) after placing predicted region in the mask

    # Calculate Dice similarity if ground truth label is provided
    dice = 0
    if lab_path is not None:
        lab_img = tio.ScalarImage(lab_path).data.squeeze()  # Load the ground truth label
        pre_img = F.interpolate(torch.from_numpy(stage2_out).float().unsqueeze(0).unsqueeze(0), size=img.data.shape[1:], mode='trilinear').float().squeeze()  # Interpolate to original space
        dice = 1 - DiceLoss()(pre_img, lab_img)  # Calculate Dice similarity
    
    # Stage 2 Output: stage2_out (label), Shape: (D, H, W) after Dice computation (if applicable)

    stage2_out = np.where(stage2_out > 0.3, 1, 0).astype(float)  # Thresholding for binary segmentation
    stage2_out_label = label(stage2_out)  # Label connected components in the binary mask
    regions = sorted(regionprops(stage2_out_label), key=lambda x: x.area, reverse=True)  # Sort regions by area
    stage2_out_img = np.where(stage2_out_label == regions[0].label, 1, 0)  # Select the largest region (bone)

    # Stage 2 Output: stage2_out_img (label), Shape: (D, H, W) after selecting the largest region

    # Interpolate the final output to original size
    out_img = F.interpolate(torch.from_numpy(stage2_out_img).float().unsqueeze(0).unsqueeze(0), size=img.data.shape[1:], mode='nearest').squeeze(0)  # Interpolate back to original space
    out_numpy = out_img.squeeze().numpy()  # Convert to numpy
    
    # Generate mesh from segmentation output
    voxel2mesh(out_numpy, img.spacing, smooth_iters=5, path=os.path.join(out_path, f'{tp}_bone.off'))

    # Save the final mask as NIfTI
    out_img = tio.ScalarImage(tensor=out_img, affine=img.affine)  # Convert to ScalarImage
    os.makedirs(out_path, exist_ok=True)  # Ensure output directory exists
    out_img.save(os.path.join(out_path, f'{tp}_bone.nii.gz'))  # Save the NIfTI file

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


    
    
    