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


from typing import Dict, Union
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
from einops import rearrange

from diffusers import (
    DiffusionPipeline,
    DDIMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)
from diffusers.utils import BaseOutput
from transformers import CLIPTextModel, CLIPTokenizer

from marigold.util.image_util import chw2hwc, colorize_depth_maps, resize_max_res
from marigold.util.batchsize import find_batch_size
from marigold.util.ensemble import ensemble_depths
from utils.loss_util import ssimloss

import torch.nn.functional as F


class MarigoldDepthOutput(BaseOutput):
    """
    Output class for Marigold monocular depth prediction pipeline.

    Args:
        depth_np (`np.ndarray`):
            Predicted depth map, with depth values in the range of [0, 1].
        depth_colored (`PIL.Image.Image`):
            Colorized depth map, with the shape of [3, H, W] and values in [0, 1].
        uncertainty (`None` or `np.ndarray`):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """

    depth_np: np.ndarray
    depth_colored: Image.Image
    uncertainty: Union[None, np.ndarray]


class MarigoldPipeline(DiffusionPipeline):
    """
    Pipeline for monocular depth estimation using Marigold: https://marigoldmonodepth.github.io.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        unet (`UNet2DConditionModel`):
            Conditional U-Net to denoise the depth latent, conditioned on image latent.
        vae (`AutoencoderKL`):
            Variational Auto-Encoder (VAE) Model to encode and decode images and depth maps
            to and from latent representations.
        scheduler (`DDIMScheduler`):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        text_encoder (`CLIPTextModel`):
            Text-encoder, for empty text embedding.
        tokenizer (`CLIPTokenizer`):
            CLIP tokenizer.
    """

    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        scheduler: DDIMScheduler,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        # inverse_scheduler: DDIMInverseScheduler

    ):
        super().__init__()

        self.register_modules(
            unet=unet,
            vae=vae,
            scheduler=scheduler,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
        )

        self.empty_text_embed = None
        self.alphas = self.scheduler.alphas_cumprod.to(self.unet.device)

    @torch.no_grad()
    def __call__(
        self,
        input_image: Image,
        denoising_steps: int = 10,
        ensemble_size: int = 10,
        processing_res: int = 768,
        match_input_res: bool = True,
        batch_size: int = 0,
        color_map: str = "Spectral",
        show_progress_bar: bool = True,
        ensemble_kwargs: Dict = None,
    ) -> MarigoldDepthOutput:
        """
        Function invoked when calling the pipeline.

        Args:
            input_image (`Image`):
                Input RGB (or gray-scale) image.
            processing_res (`int`, *optional*, defaults to `768`):
                Maximum resolution of processing.
                If set to 0: will not resize at all.
            match_input_res (`bool`, *optional*, defaults to `True`):
                Resize depth prediction to match input resolution.
                Only valid if `limit_input_res` is not None.
            denoising_steps (`int`, *optional*, defaults to `10`):
                Number of diffusion denoising steps (DDIM) during inference.
            ensemble_size (`int`, *optional*, defaults to `10`):
                Number of predictions to be ensembled.
            batch_size (`int`, *optional*, defaults to `0`):
                Inference batch size, no bigger than `num_ensemble`.
                If set to 0, the script will automatically decide the proper batch size.
            show_progress_bar (`bool`, *optional*, defaults to `True`):
                Display a progress bar of diffusion denoising.
            color_map (`str`, *optional*, defaults to `"Spectral"`):
                Colormap used to colorize the depth map.
            ensemble_kwargs (`dict`, *optional*, defaults to `None`):
                Arguments for detailed ensembling settings.
        Returns:
            `MarigoldDepthOutput`: Output class for Marigold monocular depth prediction pipeline, including:
            - **depth_np** (`np.ndarray`) Predicted depth map, with depth values in the range of [0, 1]
            - **depth_colored** (`PIL.Image.Image`) Colorized depth map, with the shape of [3, H, W] and values in [0, 1]
            - **uncertainty** (`None` or `np.ndarray`) Uncalibrated uncertainty(MAD, median absolute deviation)
                    coming from ensembling. None if `ensemble_size = 1`
        """

        device = self.device
        input_size = input_image.size

        if not match_input_res:
            assert (
                processing_res is not None
            ), "Value error: `resize_output_back` is only valid with "
        assert processing_res >= 0
        assert denoising_steps >= 1
        assert ensemble_size >= 1

        # ----------------- Image Preprocess -----------------
        # Resize image
        if processing_res > 0:
            input_image = resize_max_res(
                input_image, max_edge_resolution=processing_res
            )
        # Convert the image to RGB, to 1.remove the alpha channel 2.convert B&W to 3-channel
        input_image = input_image.convert("RGB")
        image = np.asarray(input_image)

        # Normalize rgb values
        rgb = np.transpose(image, (2, 0, 1))  # [H, W, rgb] -> [rgb, H, W]
        rgb_norm = rgb / 255.0 * 2.0 - 1.0  #  [0, 255] -> [-1, 1]
        rgb_norm = torch.from_numpy(rgb_norm).to(self.dtype)
        rgb_norm = rgb_norm.to(device)
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

        # ----------------- Predicting depth -----------------
        # Batch repeated input image
        duplicated_rgb = torch.stack([rgb_norm] * ensemble_size)
        single_rgb_dataset = TensorDataset(duplicated_rgb)
        if batch_size > 0:
            _bs = batch_size
        else:
            _bs = find_batch_size(
                ensemble_size=ensemble_size,
                input_res=max(rgb_norm.shape[1:]),
                dtype=self.dtype,
            )

        single_rgb_loader = DataLoader(
            single_rgb_dataset, batch_size=_bs, shuffle=False
        )

        # Predict depth maps (batched)
        depth_pred_ls = []
        if show_progress_bar:
            iterable = tqdm(
                single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable = single_rgb_loader
        for batch in iterable:
            (batched_img,) = batch
            depth_pred_raw = self.single_infer(
                rgb_in=batched_img,
                num_inference_steps=denoising_steps,
                show_pbar=show_progress_bar,
            )
            depth_pred_ls.append(depth_pred_raw.detach().clone())
        depth_preds = torch.concat(depth_pred_ls, axis=0).squeeze()
        torch.cuda.empty_cache()  # clear vram cache for ensembling

        # ----------------- Test-time ensembling -----------------
        if ensemble_size > 1:
            depth_pred, pred_uncert = ensemble_depths(
                depth_preds, **(ensemble_kwargs or {})
            )
        else:
            depth_pred = depth_preds
            pred_uncert = None

        # ----------------- Post processing -----------------
        # Scale prediction to [0, 1]
        min_d = torch.min(depth_pred)
        max_d = torch.max(depth_pred)
        depth_pred = (depth_pred - min_d) / (max_d - min_d)

        # Convert to numpy
        depth_pred = depth_pred.cpu().numpy().astype(np.float32)

        # Resize back to original resolution
        if match_input_res:
            pred_img = Image.fromarray(depth_pred)
            pred_img = pred_img.resize(input_size)
            depth_pred = np.asarray(pred_img)

        # Clip output range
        depth_pred = depth_pred.clip(0, 1)

        # Colorize
        depth_colored = colorize_depth_maps(
            depth_pred, 0, 1, cmap=color_map
        ).squeeze()  # [3, H, W], value in (0, 1)
        depth_colored = (depth_colored * 255).astype(np.uint8)
        depth_colored_hwc = chw2hwc(depth_colored)
        depth_colored_img = Image.fromarray(depth_colored_hwc)
        return MarigoldDepthOutput(
            depth_np=depth_pred,
            depth_colored=depth_colored_img,
            uncertainty=pred_uncert,
        )

    def __encode_empty_text(self):
        """
        Encode text embedding for empty prompt
        """
        prompt = ""
        text_inputs = self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(self.text_encoder.device)
        self.empty_text_embed = self.text_encoder(text_input_ids)[0].to(torch.float32)
    

    def single_infer_alignment(
        self, rgb_in: torch.Tensor, num_inference_steps: int, show_pbar: bool, sparse_depth, gt_depth_map=None, sparse_depth_map=None, metric=None,
        relative_structure_depth=None, inference_size=None, lr_decay_weight = 0.5, latent_lr=3e-2, pixel_lr=5e-2, latent_opt_steps=100, pixel_opt_steps=1000
    ) -> torch.Tensor:
        """
        Perform an individual depth prediction without ensembling.

        Args: 
            rgb_in (`torch.Tensor`):
                Input RGB image.
            num_inference_steps (`int`):
                Number of diffusion denoisign steps (DDIM) during inference.
            show_pbar (`bool`):
                Display a progress bar of diffusion denoising.
        Returns:
            `torch.Tensor`: Predicted depth map.
        """
        self.metric = metric
        device = rgb_in.device
        
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        time_interval = (self.scheduler.config.num_train_timesteps // len(timesteps))

        # Encode image
        rgb_latent = self.encode_rgb(rgb_in)

        # Initial depth map (noise)
        depth_latent = torch.randn(
            rgb_latent.shape, device=device, dtype=torch.float32
        )  # [B, 4, h, w]

        # Batched empty text embedding
        if self.empty_text_embed is None:
            self.__encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (rgb_latent.shape[0], 1, 1)
        )  # [B, 2, 1024]

        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)
            
        # For interpolation
        self.inference_size = inference_size    
            
        # For Structure Similarity Guidance
        self.relative_structure_depth = relative_structure_depth
        if self.relative_structure_depth is not None:
                        
            # If relative structure depth shape is different
            if self.relative_structure_depth.shape[2] != sparse_depth.shape[2] or self.relative_structure_depth.shape[3] != sparse_depth.shape[3]:
                target_size = (sparse_depth.shape[2] , sparse_depth.shape[3] )
                self.relative_structure_depth = F.interpolate(self.relative_structure_depth, size=target_size, mode='bilinear', align_corners=True, antialias=True)
        
        # FIXME
        if self.relative_structure_depth is None: 
            structure_rgb_in = (rgb_in + 1.0) / 2.0
            if self.inference_size is not None:
                self.structure_rgb_in = F.interpolate(structure_rgb_in, size=sparse_depth_map.squeeze().shape, mode='bilinear', align_corners=True, antialias=True)#.squeeze()            

        if self.relative_structure_depth is not None: self.structure_guidance = self.relative_structure_depth
        else: self.structure_guidance = self.structure_rgb_in

        count = -1
        for i, t in iterable:            
            total_steps = len(timesteps)
            index = total_steps-i-1

            # Concat the each modality
            unet_input = torch.cat(
                [rgb_latent, depth_latent], dim=1
            )  # this order is important

            # Predict the noise residual
            noise_pred = self.unet(
                unet_input, t, encoder_hidden_states=batch_empty_text_embed
            ).sample  # [B, 4, h, w]

            # Compute the t=0 sample z_0 
            depth_latent_zt = self.scheduler.step(noise_pred, t, depth_latent)["prev_sample"]
            depth_latent = depth_latent_zt.detach()
            
            # Tweedie's forumal for computing \hat{z}_0 at timestep=t
            sqrt_one_minus_alpha_prod = (1 - self.scheduler.alphas_cumprod[t]) ** 0.5
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten().to(device)
            
            depth_latent_pseudo_z0 = (depth_latent - sqrt_one_minus_alpha_prod**2 * noise_pred) / (self.scheduler.alphas_cumprod[t].to(device))**0.5
            
            # Optimization loop configuration
            splits = 3 # Split the total infrence sampling process into {splits} parts
            total_steps = len(timesteps)
            index_split = total_steps // splits            
            optimization_interval = 5
            
            # Do optimization loop parts (second & third parts)
            if index <= (total_steps - index_split) and index>0 :
                z_t = depth_latent_zt.detach().clone()
                
                if index % optimization_interval == 0 :  
                    time = t
                    
                    # Obtain z_{t-k}
                    unet_input = torch.cat(
                        [rgb_latent, z_t], dim=1
                    )  # this order is important

                    # predict the noise residual
                    noise_pred = self.unet(
                        unet_input, time, encoder_hidden_states=batch_empty_text_embed
                    ).sample  # [B, 4, h, w]
                    
                    # Compute the t=t sample z_t
                    depth_latent_zt = self.scheduler.step(noise_pred, time, z_t)["prev_sample"]
                    
                    # Tweedie's forumal for computing \hat{z}_0
                    sqrt_one_minus_alpha_prod = (1 - self.scheduler.alphas_cumprod[time]) ** 0.5
                    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten().to(device)
                    
                    depth_latent_pseudo_z0 = (z_t - sqrt_one_minus_alpha_prod**2 * noise_pred) / (self.scheduler.alphas_cumprod[time].to(device))**0.5
                        
                    if index >= 0:
                        sigma = 40*(1-self.scheduler.alphas_cumprod[time-time_interval].to(device)) / (1-self.scheduler.alphas_cumprod[time].to(device)) * (1- self.scheduler.alphas_cumprod[time].to(device)/self.scheduler.alphas_cumprod[time-time_interval].to(device))
                    else:
                        sigma=0.5

                    # First part: pixel-based optimization loop (opt var = \hat{x_0} = D(\hat{z}_0))
                    if index >= index_split: 
                        depth_latent_pseudo_z0 = depth_latent_pseudo_z0.detach()
                        depth_pseudo_x0 = self.decode_depth(depth_latent_pseudo_z0, )#cnn=True)
                                                
                        if self.inference_size is not None:
                            depth_pseudo_x0 = F.interpolate(depth_pseudo_x0, size=sparse_depth_map.squeeze().shape, mode='bilinear', align_corners=True, antialias=True)#ANCHOR - .squeeze()
                        
                        depth_pseudo_x0 = torch.clip(depth_pseudo_x0, -1.0, 1.0)
                        depth_pseudo_x0 = (depth_pseudo_x0 + 1.0) / 2.0 # 0 to 1
                                  
                        # Pixel optimization loop              
                        opt_var = self.pixel_optimization(depth_pseudo_x0, sparse_depth, max_iters=pixel_opt_steps, lr = pixel_lr)#, edge_guidance, affinity_gt)
                        
                        # FIXME
                        if self.inference_size is not None:
                            opt_var = F.interpolate(opt_var, size=rgb_in.shape[-2:], mode='bilinear', align_corners=True, antialias=True)#.squeeze()
                        
                        depth_opt_z0 = self.encode_depth(opt_var)
                        
                        # Resample
                        depth_latent_zt = self.stochastic_resample(depth_opt_z0, z_t, self.scheduler.alphas_cumprod[time-time_interval].to(device), sigma , device)
                        # depth_latent_zt = self.stochastic_encode(opt_var, self.scheduler.alphas_cumprod[time-time_interval].to(device), sigma , device)
                        
                        depth_latent_zt = depth_latent_zt.requires_grad_(True)
                        
                    # Second part: latent-based optimization loop (opt var = \hat{z_0})
                    elif index < index_split:
                        # Learning rate decay
                        count += 1
                        latent_lr_local = latent_lr*(lr_decay_weight**count)
                        
                        # Latent optimization loop
                        depth_latent_pseudo_z0 = self.latent_optimization(depth_latent_pseudo_z0.detach(),sparse_depth, gt_depth_map, sparse_depth_map, max_iters=latent_opt_steps, lr = latent_lr_local)
                
                        # Resample
                        sigma = 40 * (1-self.scheduler.alphas_cumprod[time-time_interval].to(device))/(1 - self.scheduler.alphas_cumprod[time].to(device)) * (1 - self.scheduler.alphas_cumprod[time].to(device) / self.scheduler.alphas_cumprod[time-time_interval].to(device)) # Change the 40 value for each task
                        depth_latent_zt = self.stochastic_resample(pseudo_x0=depth_latent_pseudo_z0, x_t=z_t, a_t=self.scheduler.alphas_cumprod[time-time_interval].to(device), sigma=sigma, device=device) 
                        # depth_latent_zt = self.stochastic_encode(pseudo_x0=depth_latent_pseudo_x0, a_t=self.scheduler.alphas_cumprod[time-time_interval].to(device), sigma=sigma, device=device) 
                                            
                    depth_latent = depth_latent_zt.detach()


        # Last optimization loop at final sampling step
        pseudo_z0 = self.latent_optimization(depth_latent_zt.detach(),sparse_depth,gt_depth_map, sparse_depth_map, max_iters=latent_opt_steps, lr=latent_lr_local)#,edge_guidance, affinity_gt)# max_iters=5)
        depth_latent_zt = self.stochastic_resample(pseudo_x0=pseudo_z0, x_t=z_t, a_t=self.scheduler.alphas_cumprod[time-time_interval].to(device), sigma=sigma, device=device) 
        # depth_latent_zt = self.stochastic_encode(pseudo_z0=pseudo_z0, a_t=self.scheduler.alphas_cumprod[time-time_interval].to(device), sigma=sigma, device=device) 
        
        unet_input = torch.cat(
                [rgb_latent, depth_latent_zt.detach()], dim=1
        )  # this order is important

        # predict the noise residual
        noise_pred = self.unet(
            unet_input, 1, encoder_hidden_states=batch_empty_text_embed
        ).sample  # [B, 4, h, w]

        depth_latent = self.scheduler.step(noise_pred, 1, depth_latent).prev_sample
        
        return depth_latent

    def stochastic_resample(self, pseudo_x0, x_t, a_t, sigma, device):
        """
        Function to resample x_t based on ReSample paper.
        """
        noise = torch.randn_like(pseudo_x0, device=device)
        return (sigma * a_t.sqrt() * pseudo_x0 + (1 - a_t) * x_t)/(sigma + 1 - a_t) + noise * torch.sqrt(1/(1/sigma + 1/(1-a_t)))


    def stochastic_encode(self, pseudo_x0, a_t, sigma, device):
        noise = torch.randn_like(pseudo_x0, device=device)
        return a_t.sqrt() * pseudo_x0 + (1-a_t) * noise 

    def pixel_optimization(self, x_prime, sparse_depth, lr=5e-2, max_iters=1000): # x_prime mean \hat(x0)
        measurement_loss = torch.nn.L1Loss() # or MSELoss() or L1Loss()

        opt_var = x_prime.detach().clone()
        opt_var = opt_var.requires_grad_()
        optimizer = torch.optim.AdamW([opt_var], lr=lr) 
        sparse_depth = sparse_depth.detach()

        # Training loop
        mask = sparse_depth > 0 # -1

        for _ in tqdm(range(max_iters)):
            optimizer.zero_grad()
            
            l_measurement = measurement_loss(sparse_depth[mask], opt_var[mask] ) 
            l_local_smoothness = self.local_smoothness_loss(self.structure_guidance,opt_var)* 0.1

            # if self.structure_guidance.shape[1] == 3: 
            #     l_ssim = (1- ssimloss(opt_var, self.structure_guidance[:,1,...])) *0.1
            # else:
            #     l_ssim = (1- ssimloss(opt_var, self.structure_guidance)) *0.1
                                    
            loss = l_measurement + l_local_smoothness # + l_ssim
            loss.backward()
            optimizer.step()
        
        return opt_var
        
    def latent_optimization(self, z_init, sparse_depth, gt_depth_map=None, sparse_depth_map=None, max_iters=200, lr=3e-2):
        # Base case
        if not z_init.requires_grad:
            z_init = z_init.requires_grad_()
            
        measurement_loss = torch.nn.L1Loss()
        optimizer = torch.optim.AdamW([z_init], lr=lr) 
        sparse_depth = sparse_depth.detach()
        
        mask = sparse_depth > 0 

        for _ in tqdm(range(max_iters)):
            optimizer.zero_grad()
            depth = self.decode_depth(z_init)
            
            if self.inference_size is not None:
                depth = F.interpolate(depth, size=sparse_depth_map.squeeze().shape, mode='bilinear', align_corners=True, antialias=True)#.squeeze()
            # clip prediction
            depth = torch.clip(depth, -1.0, 1.0)
            # shift to [0, 1]
            depth = (depth + 1.0) / 2.0
            
            l_measurement = measurement_loss(sparse_depth[mask], depth[mask] )
            
            # you need to manipulate each regularizaiton loss weight term.
            l_local_smoothness = self.local_smoothness_loss(self.structure_guidance,depth) *0.1
            
            if self.structure_guidance.shape[1] == 3: 
                l_ssim = (1- ssimloss(depth, self.structure_guidance[:,1,...])) *0.1
            else:
                l_ssim = (1- ssimloss(depth, self.structure_guidance)) *0.1

            loss = l_measurement + l_local_smoothness + l_ssim
                
            loss.backward()
            optimizer.step()

            self.metric.evaluate(depth.squeeze().detach().cpu().numpy(), gt_depth_map.squeeze(), sparse_depth_map)

        return z_init

    @torch.no_grad()
    def single_infer(
        self, rgb_in: torch.Tensor, num_inference_steps: int, show_pbar: bool
    ) -> torch.Tensor:
        """
        Perform an individual depth prediction without ensembling.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image.
            num_inference_steps (`int`):
                Number of diffusion denoisign steps (DDIM) during inference.
            show_pbar (`bool`):
                Display a progress bar of diffusion denoising.
        Returns:
            `torch.Tensor`: Predicted depth map.
        """
        # with torch.no_grad():
        device = rgb_in.device
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps  # [T]

        # Encode image
        rgb_latent = self.encode_rgb(rgb_in)

        
        # Initial depth map (noise)
        depth_latent = torch.randn(
            rgb_latent.shape, device=device, dtype=self.dtype
        )  # [B, 4, h, w]

        # Batched empty text embedding
        if self.empty_text_embed is None:
            self.__encode_empty_text()
        batch_empty_text_embed = self.empty_text_embed.repeat(
            (rgb_latent.shape[0], 1, 1)
        )  # [B, 2, 1024]

        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)

        for i, t in iterable:
            # with torch.no_grad():
            unet_input = torch.cat(
                [rgb_latent, depth_latent], dim=1
            )  # this order is important
                
            torch.cuda.empty_cache()

            # predict the noise residual
            noise_pred = self.unet(
                unet_input, t, encoder_hidden_states=batch_empty_text_embed
            ).sample  # [B, 4, h, w]

            # compute the previous noisy sample x_t -> x_t-1
            depth_latent = self.scheduler.step(noise_pred, t, depth_latent).prev_sample.detach()
            
        depth = self.decode_depth(depth_latent)

        # clip prediction
        depth = torch.clip(depth, -1.0, 1.0)
        # # shift to [0, 1]
        depth = (depth + 1.0) / 2.0
            
        return depth
    

    def encode_rgb(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """
        # encode        
        h = self.vae.encoder(rgb_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)    
            
        # scale latent
        rgb_latent = mean * self.rgb_latent_scale_factor
        return rgb_latent


    def decode_depth(self, depth_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (`torch.Tensor`):
                Depth latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """
        # scale latent
        depth_latent = depth_latent / self.depth_latent_scale_factor
        
        # decode
        z = self.vae.post_quant_conv(depth_latent)
        stacked = self.vae.decoder(z)
        
        # mean of output channels
        depth_mean = stacked.mean(dim=1, keepdim=True)
        return depth_mean

    def normalize_depth(self, depth_in: torch.Tensor, lower_percentile: float = 0.02 , upper_percentile: float = 0.98 ) -> torch.Tensor:
        """
        Normalize depth image into latent.  => Individual depth map normalizatoin

        Args:
            depth_in (`torch.Tensor`):
                Input depth image to be encoded.
                (B, 1, H, W)

        Returns:
            `torch.Tensor`: normalized depth.
        """
        
        falttened_depth = rearrange(depth_in, 'b c h w -> b (c h w)') #depth_in.flatten()
                
        lower_value = torch.quantile(falttened_depth.type(torch.FloatTensor), lower_percentile, dim=1, keepdim=True).to(falttened_depth.device)
        # print(lower_value)
        # print(lower_value.shape)
        upper_value = torch.quantile(falttened_depth.type(torch.FloatTensor), upper_percentile, dim=1, keepdim=True).to(falttened_depth.device)
        # print(upper_value)
        # print(upper_value.shape)
        depth_norm = torch.clamp((((falttened_depth - lower_value) / (upper_value - lower_value))-0.5)*2, -1, 1)
    
        depth_norm = rearrange(depth_norm, 'b (c h w) -> b c h w', c=1, h=depth_in.shape[2], w=depth_in.shape[3])
    
        depth_norm = depth_norm.to(depth_in.device)#.type(torch.HalfTensor).to(depth_in.device)
    
        return depth_norm
    
    def encode_depth(self, depth_in: torch.Tensor) -> torch.Tensor: 
        """
        Encode depth image into latent. 

        Args:
            depth_in (`torch.Tensor`):
                Input depth image to be encoded.
                (B, 1, H, W)

        Returns:
            `torch.Tensor`: Image latent.
        """
        norm_depth_in = depth_in # depth is normalized depth
        norm_depth_in = torch.cat([norm_depth_in] * 3, dim=1)
        # encode
        h = self.vae.encoder(norm_depth_in)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        # scale latent
        depth_latent = mean * self.depth_latent_scale_factor
        return depth_latent
    
    def local_smoothness_loss(self, image, depth_map):
        # 이미지 말고, 첫 depth map 예측한거의 GRAdient를 guide로 주면 어떨까?
        image = torch.mean(image, dim=1, keepdim=True)
        
        # Calculate gradient ogradient_x = F.conv2d(depth_map, torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3).to(depth_map.device))
        gradient_x_d = F.conv2d(depth_map, torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3).to(depth_map.device))
        gradient_y_d = F.conv2d(depth_map, torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3).to(depth_map.device))        
        
        # Calculate gradient of depth
        gradient_x = F.conv2d(image, torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3).to(image.device))
        gradient_y = F.conv2d(image, torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3).to(image.device)) 
        
        exp_gradient_x = torch.exp(-torch.abs(gradient_x))
        exp_gradient_y = torch.exp(-torch.abs(gradient_y))
        
        # Weighted smoothing
        smoothed_gradient_x = torch.abs(gradient_x_d) * exp_gradient_x
        smoothed_gradient_y = torch.abs(gradient_y_d) * exp_gradient_y
        
        # Combine gradients
        smoothed_loss = torch.mean(smoothed_gradient_x) + torch.mean(smoothed_gradient_y)
        
        return smoothed_loss
