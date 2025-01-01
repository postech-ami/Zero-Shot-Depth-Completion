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


# NOTE: This script is a just example of obtaining structured realtive depth map
# You can use Marigold-LCM, DepthFM, or any other depth foundation model to obtain structured affine-invariant depth map 

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from tqdm.auto import tqdm

from marigold import MarigoldPipeline
from marigold.util.ensemble import ensemble_depths

from utils.general_util import set_seed
from utils.depth_completion_util import normalize_rgb


if "__main__" == __name__:
    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image depth estimation using Marigold."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="prs-eth/marigold-v1-0",
        help="Checkpoint path or hub name.",
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
        "--n_sampling_steps",
        type=int,
        default=50,
        help="Fix random seed.",
    )
    
    parser.add_argument(
        "--n_ensemble",
        type=int,
        default=3,
        help="Fix random seed.",
    )
    
    parser.add_argument(
        "--input_root_dir",
        type=str,
        required=True,
        help="Path to the input folder, where rgb, sparse depth, and gt depth images are stored.",
    )

    args = parser.parse_args()

    # Base model setting
    depth_diffusion = MarigoldPipeline.from_pretrained(args.checkpoint, torch_dtype=torch.float32 ).to(args.device)
    
    # Set seed
    set_seed(args.seed)

    # Read information
    for filename in os.listdir(args.input_root_dir):
        if "rgb" in filename: rgb_path = os.path.join(args.input_root_dir, filename)
    
    rgb_img = Image.open(rgb_path).convert("RGB")

    # FIXME - for 104, and adjust it for void
    from torchvision import transforms
    transform = transforms.Compose([
            transforms.PILToTensor(),
            transforms.CenterCrop((352, 1216)),
        ])
    
    rgb_img = transform(rgb_img).unsqueeze(0)
    norm_rgb = normalize_rgb(rgb_img).to(torch.float32).to(args.device)
    
    with torch.autocast('cuda', dtype=torch.float32):
        depth_list = []
        for _ in tqdm(range(args.n_ensemble)):
            
            single_marigold_depth = depth_diffusion.single_infer(
                rgb_in=norm_rgb,#.type('torch.HalfTensor').to(device),
                num_inference_steps=args.n_sampling_steps,
                show_pbar=True,
            )
            
            depth_list.append(single_marigold_depth.detach())
            torch.cuda.empty_cache()

        depth = torch.cat(depth_list,dim=0).squeeze()
        torch.cuda.empty_cache()        
        
        # Ensemble
        ensemble_kwargs =None
        marigold_depth, pred_uncert = ensemble_depths(
                depth, **(ensemble_kwargs or {})
            )
        marigold_depth = marigold_depth.unsqueeze(0).unsqueeze(0).to(torch.float32).cpu()
                
        torch.save(marigold_depth, os.path.join(args.input_root_dir, "marigold_depth.pt"))            
    