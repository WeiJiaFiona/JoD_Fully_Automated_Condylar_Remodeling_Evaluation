
import scipy.ndimage as si
import torch 
import numpy as np 

def gaussian_map(img, center, sigma):

    img[center] = 1
    img = si.gaussian_filter(img, sigma=sigma)
    
    return img

def crop_3d_centroid(img, centroid, patch_size):
    '''
        Crop the 3D image into patches
        @params:
            img: Original 3D image
            centroid: centroid of one Tooth
            patch_size: cropped patch size
        @returns:
            Cropped patch
    '''
    assert len(img.shape) == 3 and len(centroid) == 3
    
    # obtain the img size
    H, W, D = img.shape
    
    # obtain the patch size
    if not isinstance(patch_size, (list, tuple, np.ndarray)):
        patch_size = [patch_size] * 3
    
    # obtain the crop size
    h_crop_1 = centroid[0] - patch_size[0] // 2
    h_crop_2 = centroid[0] + patch_size[0] - patch_size[0] // 2

    w_crop_1 = centroid[1] - patch_size[1] // 2
    w_crop_2 = centroid[1] + patch_size[1] - patch_size[1] // 2

    d_crop_1 = centroid[2] - patch_size[2] // 2
    d_crop_2 = centroid[2] + patch_size[2] - patch_size[2] // 2    
    
    # obtain the padding size 
    h_pad_1 = 0 - h_crop_1 if h_crop_1 < 0 else 0
    h_pad_2 = h_crop_2 - H if h_crop_2 > H else 0
    
    w_pad_1 = 0 - w_crop_1 if w_crop_1 < 0 else 0
    w_pad_2 = w_crop_2 - W if w_crop_2 > W else 0

    d_pad_1 = 0 - d_crop_1 if d_crop_1 < 0 else 0
    d_pad_2 = d_crop_2 - D if d_crop_2 > D else 0 
    
    # correct the crop size
    h_crop_1 = 0 if h_crop_1 < 0 else h_crop_1
    h_crop_2 = H if h_crop_2 < 0 else h_crop_2

    w_crop_1 = 0 if w_crop_1 < 0 else w_crop_1
    w_crop_2 = W if w_crop_2 < 0 else w_crop_2

    d_crop_1 = 0 if d_crop_1 < 0 else d_crop_1
    d_crop_2 = D if d_crop_2 < 0 else d_crop_2
    
    patch_img = img[h_crop_1:h_crop_2, w_crop_1:w_crop_2, d_crop_1:d_crop_2]
    pad_img = np.pad(patch_img, ((h_pad_1, h_pad_2), (w_pad_1, w_pad_2), (d_pad_1, d_pad_2)), 'constant')
    
    return pad_img


def collate_fn(batch):
    collate_batch = {}
    for k in batch[0].keys():
        if isinstance(batch[0][k], (torch.LongTensor, torch.FloatTensor)):
            collate_batch[k] = torch.stack([item for item in map(lambda x:x[k], batch)], dim=0)
        else:
            collate_batch[k] = [item for item in map(lambda x:x[k], batch)]

    return collate_batch

def center_shift_crop_pad(img, centers, sigma, crop_size, shift_range=12):
    crop_hmaps = []
    crop_imgs = []
    shift_centers = []
    pad_img = np.pad(img, ((crop_size[0]//2, crop_size[0] - crop_size[0]//2), (crop_size[1]//2, crop_size[1] - crop_size[1]//2), (crop_size[0]//2, crop_size[0] - crop_size[0]//2), ), 'constant')
    for img_center in centers:
        if shift_range > 0:
            shift = np.array([np.random.randint(-shift_range, shift_range), np.random.randint(-shift_range, shift_range), np.random.randint(-shift_range, shift_range)])
        else:
            shift = np.array([0, 0, 0])
        hmap_center = (np.array(crop_size) / 2 + shift).astype(int)
        center = (np.array(img_center) + shift).astype(int)
        crop_hmap = gaussian_map(np.zeros(crop_size), tuple(hmap_center), sigma)
        crop_img = crop_3d_centroid(img, center, crop_size)
    
        crop_hmaps.append(crop_hmap)
        crop_imgs.append(crop_img)
        shift_centers.append(hmap_center)
    
    return np.stack(crop_imgs, axis=0), np.stack(crop_hmaps, axis=0), np.stack(shift_centers, axis=0)
        