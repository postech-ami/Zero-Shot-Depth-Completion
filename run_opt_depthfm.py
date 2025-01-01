# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from tqdm.auto import tqdm

from depthfm import DepthFM

from utils.general_util import set_seed
from utils.depth_completion_util import normalize_rgb, normalize_sparse_depth, DepthCompletionMetric

if "__main__" == __name__:
    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image depth estimation using Marigold."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/node_data/hyoseok/checkpoints/depthfm-v1.ckpt",
        help="Checkpoint path of depthfm",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on.",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Fix random seed.",
    )
    
    parser.add_argument(
        "--input_root_dir",
        type=str,
        required=True,
        help="Path to the input folder, where rgb, sparse depth, and gt depth images are stored.",
    )

    parser.add_argument(
        "--output_dir", type=str, default=None, help="Output directory."
    )
    
    parser.add_argument(
        "--r_ssim_depth", action='store_true', help="For r-ssim, use depth map instead of rgb image."
    )
    
    parser.add_argument(
    "--inference_size", type=tuple, default=(512, 512), help="Inference depth size as (width, height)."
    )   
    
    parser.add_argument(
        "--data_type",
        type=str,
        choices=["indoor", "outdoor"],
        default="outdoor",
        help="Specify the evaluation type: 'indoor' or 'outdoor'."
    )

    parser.add_argument("--num_steps", type=int, default=4,
                        help="Number of steps for ODE solver")
    
    parser.add_argument("--n_inter", type=int, default=0,
                        help="intermediate step estimation. Default is 0, which means one-step generation. If set to 1, it will generate two-step depth map.")

    args = parser.parse_args()
    
    metric = DepthCompletionMetric(data_type=args.data_type)

    # Base model setting
    depth_diffusion = DepthFM(args.checkpoint, metric = metric).to(args.device)
    
    # Set seed
    set_seed(args.seed)

    # Set output directory: default is "{input_directory}/outputs"
    if args.output_dir is not None: output_dir = args.output_dir
    else: output_dir = os.path.join(args.input_root_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    # Read information
    for filename in os.listdir(args.input_root_dir):
        if "rgb" in filename: rgb_path = os.path.join(args.input_root_dir, filename)
        if "sparse" in filename: sparse_path = os.path.join(args.input_root_dir, filename)
        if "gt" in filename: gt_path = os.path.join(args.input_root_dir, filename)
    
    rgb_img = Image.open(rgb_path).convert("RGB")
    sparse_depth_map = Image.open(sparse_path)
    gt_depth_map = Image.open(gt_path)
    
    # FIXME - for 104, and adjust it for void
    from torchvision import transforms
    transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.CenterCrop((352, 1216)),
        ])
    
    # rgb_img = transform(rgb_img).unsqueeze(0)[..., 104:, :]
    # sparse_depth_map = transform(sparse_depth_map).unsqueeze(0)[..., 104:, :] / 256.
    # gt_depth_map = transform(gt_depth_map).unsqueeze(0)[..., 104:, :] / 256.
    
    if args.inference_size is not None: rgb_img = transforms.Resize(args.inference_size)(transform(rgb_img)).unsqueeze(0)
    else: rgb_img = transform(rgb_img).unsqueeze(0)
    sparse_depth_map = transform(sparse_depth_map).unsqueeze(0) / 256.
    gt_depth_map = transform(gt_depth_map).unsqueeze(0) / 256.
    
    if args.r_ssim_depth:
        relative_structure_depth = torch.load(os.path.join(args.input_root_dir, "depthfm_depth.pt"), map_location=args.device).to(torch.float32)
    else:
        relative_structure_depth = None

    gt_mask = gt_depth_map>0
    sparse_mask = sparse_depth_map>1e-8
        
    # Scale prediction to [0, 1]
    sparse_depth_map = sparse_depth_map.to(args.device)

    norm_sparse_depth = normalize_sparse_depth(sparse_depth_map)    
    norm_rgb = normalize_rgb(rgb_img).to(torch.float32).to(args.device)
    
    metric = DepthCompletionMetric(data_type=args.data_type)
    with torch.autocast(device_type="cuda", dtype=torch.float32):
        # depth = model.predict_depth(im, num_steps=args.num_steps, ensemble_size=args.ensemble_size)
        depth = depth_diffusion.forward(norm_rgb, num_steps=args.num_steps, ensemble_size=1, norm_sparse_depth=norm_sparse_depth, sparse_depth_map=sparse_depth_map, gt_depth_map=gt_depth_map, n_intermediates=args.n_inter,
                                        relative_structure_depth=relative_structure_depth, inference_size=args.inference_size)
    
    depth = F.interpolate(depth, size=sparse_depth_map.squeeze().shape, mode='bilinear', align_corners=True, antialias=True)
            
    metrics = metric.evaluate(depth.squeeze().detach().cpu().numpy(), gt_depth_map.squeeze(), sparse_depth_map.squeeze())
    
    # Least square fitting                
    pred = depth.squeeze().detach().cpu().numpy()
    gt = sparse_depth_map.squeeze().detach().cpu().numpy()
    
    mask = gt > 0
    num_valid = mask.sum()
    
    sparse_depth = sparse_depth_map.squeeze().detach().cpu().numpy()
    mask = sparse_depth>0.
    min_sparse = np.min(sparse_depth[mask])
    max_sparse = np.max(sparse_depth[mask])
    
    pred = pred * (max_sparse - min_sparse) + min_sparse
    a,b = np.polyfit(pred[mask], gt[mask], deg=1)

    if a > 0:
        pred = a * pred + b
    pred = np.clip(pred, 0, 80)
    
    # Save raw completed depth map
    pred = (pred*256).astype(np.uint16)
    pred_raw = Image.fromarray(pred)
    pred_raw.save(os.path.join(output_dir, "depthfm_pred_raw.png"))
    
    # Save colorized depth map
    cmap = 'jet'
    cm = plt.get_cmap(cmap)
    
    depth_color = depth.squeeze().detach().cpu().numpy()
    depth_color = cm(depth_color)
    plt.imsave(os.path.join(output_dir, "depthfm_pred_color.png"), depth_color)

    