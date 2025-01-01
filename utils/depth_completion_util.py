import torch
import numpy as np

def get_sparse_depth( dep, num_sample=60000):
    channel, height, width = dep.shape

    assert channel == 1

    idx_nnz = torch.nonzero(dep.view(-1) > 0.0001, as_tuple=False)

    num_idx = len(idx_nnz)
    idx_sample = torch.randperm(num_idx)[:num_sample]

    idx_nnz = idx_nnz[idx_sample[:]]

    mask = torch.zeros((channel*height*width))
    mask[idx_nnz] = 1.0
    mask = mask.view((channel, height, width))
    
    mask_far = dep > 0.9375
    
    mask[mask_far] = 0.0

    dep_sp = dep * mask.type_as(dep)

    return dep_sp

def normalize_rgb(rgb_in: torch.Tensor) -> torch.Tensor :
    # [ B, 3, H, W ]
    
    # Normalize rgb 
    rgb_norm = (rgb_in / 255.0 ) * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
    rgb_norm = rgb_norm#.to(self.device)
    assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0
    
    return rgb_norm

def normalize_sparse_depth(depth_in: torch.Tensor) -> torch.Tensor :
    # [ B, 3, H, W ]
    mask = depth_in > 0.
    # Normalize rgb 
    min_d = torch.min(depth_in[mask])
    max_d = torch.max(depth_in[mask])

    depth_norm = ((depth_in - min_d) / (max_d - min_d))  #  [0, 255] -> [0, 1]
    
    return depth_norm

import os
import numpy as np

import torch
from PIL import Image


class DepthCompletionMetric():
    def __init__(self, data_type='outdoor'):
        super(DepthCompletionMetric, self)
        self.t_valid = 0.0001   # 0.00

        self.metric_name = [
            'RMSE', 'MAE', 'iRMSE', 'iMAE', 'REL', 'D^1', 'D^2', 'D^3', 'D102', 'D105', 'D110'
        ]
        
        self.data_type= data_type


    def evaluate(self, path_pred, path_gt, sparse_depth):
        with torch.no_grad():
            pred = path_pred.squeeze()
            gt = path_gt.squeeze().detach().cpu().numpy()
            sparse_depth = sparse_depth.squeeze().detach().cpu().numpy()

            # sparse
            sparse_mask = sparse_depth > self.t_valid
            
            # https://github.com/seobbro/TTA-depth-completion
            # For same evaluation, we set the valid range of depth values to be the same with the above paper.
            # if self.data_type == 'indoor':
            #     min_max_mask =  np.logical_and(
            #                             gt > 0.2,
            #                             gt < 5)
                                        
            #     sparse_mask = np.where(np.logical_and(sparse_mask, min_max_mask)>0)
            # elif self.data_type == 'outdoor':  
            #     min_max_mask =  np.logical_and(
            #                             gt > 0,
            #                             gt < 80)
                                        
            #     sparse_mask = np.where(np.logical_and(sparse_mask, min_max_mask)>0)
        
            # Least-sqaure fitting for finding the scale and shift factors    
            a,b = np.polyfit(pred[sparse_mask], sparse_depth[sparse_mask], deg=1)

            if a > 0:
                pred = a * pred + b


            pred_inv = 1.0 / (pred + 1e-8)
            gt_inv = 1.0 / (gt + 1e-8)

            # https://github.com/seobbro/TTA-depth-completion
            # For same evaluation, we set the valid range of depth values to be the same with the above paper.
            gt_mask = gt > self.t_valid

            if self.data_type == 'indoor':  
                min_max_mask =  np.logical_and(
                                        gt > 0.2,
                                        gt < 5)
                                        
                gt_mask = np.logical_and(gt_mask, min_max_mask)
            elif self.data_type == 'outdoor':  
                min_max_mask =  np.logical_and(
                                        gt > 0,
                                        gt < 80)
                                        
                gt_mask = np.logical_and(gt_mask, min_max_mask)            
            
            num_valid = gt_mask.sum()

            pred = torch.from_numpy(pred[gt_mask])
            gt = torch.from_numpy(gt[gt_mask])

            pred_inv = torch.from_numpy(pred_inv[gt_mask])
            gt_inv = torch.from_numpy(gt_inv[gt_mask])

            pred_inv[pred <= self.t_valid] = 0.0
            gt_inv[gt <= self.t_valid] = 0.0

            # RMSE / MAE
            diff = pred - gt
            diff_abs = torch.abs(diff)
            diff_sqr = torch.pow(diff, 2)

            rmse = diff_sqr.sum() / (num_valid + 1e-8)
            rmse = torch.sqrt(rmse)

            mae = diff_abs.sum() / (num_valid + 1e-8)

            # iRMSE / iMAE
            diff_inv = pred_inv - gt_inv
            diff_inv_abs = torch.abs(diff_inv)
            diff_inv_sqr = torch.pow(diff_inv, 2)

            irmse = diff_inv_sqr.sum() / (num_valid + 1e-8)
            irmse = torch.sqrt(irmse)

            imae = diff_inv_abs.sum() / (num_valid + 1e-8)

            # Rel
            rel = diff_abs / (gt + 1e-8)
            rel = rel.sum() / (num_valid + 1e-8)

            # delta
            r1 = gt / (pred + 1e-8)
            r2 = pred / (gt + 1e-8)
            ratio = torch.max(r1, r2)

            del_1 = (ratio < 1.25).type_as(ratio)
            del_2 = (ratio < 1.25**2).type_as(ratio)
            del_3 = (ratio < 1.25**3).type_as(ratio)
            del_102 = (ratio < 1.02).type_as(ratio)
            del_105 = (ratio < 1.05).type_as(ratio)
            del_110 = (ratio < 1.10).type_as(ratio)

            del_1 = del_1.sum() / (num_valid + 1e-8)
            del_2 = del_2.sum() / (num_valid + 1e-8)
            del_3 = del_3.sum() / (num_valid + 1e-8)
            del_102 = del_102.sum() / (num_valid + 1e-8)
            del_105 = del_105.sum() / (num_valid + 1e-8)
            del_110 = del_110.sum() / (num_valid + 1e-8)

            result = [rmse, mae, irmse, imae, rel, del_1, del_2, del_3, del_102, del_105, del_110]

        print(result)
        return result
