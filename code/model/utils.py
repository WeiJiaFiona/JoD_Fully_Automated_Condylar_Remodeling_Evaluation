import torch
import torch.nn as nn
import torch.nn.functional as F

def _nms(heat, kernel=3):
    hmax = F.max_pool3d(heat, kernel, stride=1, padding=(kernel - 1) // 2)
    keep = (hmax == heat).float()
    return heat * keep

def _topk(scores, K=1):

    B, C, H, W, D = scores.size()

    # look through the topk inds for Batch and Categories
    
    topk_scores, topk_inds = torch.topk(scores.view(B, C, -1), K)

    topk_inds = topk_inds % (H * W * D)
    topk_zs = (topk_inds % (W*D) % D).int().float()
    topk_ys = (topk_inds % (W*D) / D).int().float()
    topk_xs = (topk_inds / (W*D)).int().float()
    

    return topk_scores, torch.stack([topk_xs, topk_ys, topk_zs], dim=-1)



def decode_hmap(hmap):
    
    hmap=torch.sigmoid(hmap)
    hmap = _nms(hmap)  # perform nms on heatmaps
    scores, coords = _topk(hmap)  # get topk scores, coords
    
    return hmap, coords
    
    

    
    
    
    