import torch
import einops
import numpy as np
import torch.nn as nn
from torch import Tensor
from functools import partial
from torchdiffeq import odeint

import torch.nn.functional as F

from tqdm.auto import tqdm

from unet import UNetModel
from diffusers import AutoencoderKL

from utils.loss_util import ssimloss

class DepthFM(nn.Module):
    def __init__(self, ckpt_path: str, metric=None):
        super().__init__()
        vae_id = "runwayml/stable-diffusion-v1-5"
        self.vae = AutoencoderKL.from_pretrained(vae_id, subfolder="vae")
        self.scale_factor = 0.18215
        
        self.metric = metric

        # set with checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.noising_step = ckpt['noising_step']
        self.empty_text_embed = ckpt['empty_text_embedding']
        self.model = UNetModel(**ckpt['ldm_hparams'])
        self.model.load_state_dict(ckpt['state_dict'])
    
    def ode_fn(self, t: Tensor, x: Tensor, **kwargs):
        if t.numel() == 1:
            t = t.expand(x.size(0))
        return self.model(x=x, t=t, **kwargs)
    
    def generate(self, z: Tensor, num_steps: int = 4, n_intermediates: int = 0, **kwargs):
        """
        ODE solving from z0 (ims) to z1 (depth).
        """
        ode_kwargs = dict(method="euler", rtol=1e-5, atol=1e-5, options=dict(step_size=1.0 / num_steps))
        
        # t specifies which intermediate times should the solver return
        # e.g. t = [0, 0.5, 1] means return the solution at t=0, t=0.5 and t=1
        # but it also specifies the number of steps for fixed step size methods
        t = torch.linspace(0, 1, n_intermediates + 2, device=z.device, dtype=z.dtype)

        # allow conditioning information for model
        ode_fn = partial(self.ode_fn, **kwargs)
        
        ode_results = odeint(ode_fn, z, t, **ode_kwargs)
        
        if n_intermediates > 0:
            return ode_results
        return ode_results[-1]
    
    def generate_alignment_one_step(self, z: Tensor, num_steps: int = 4, n_intermediates: int = 0, img=None, sparse_depth_map=None, norm_sparse_depth=None, gt_depth_map=None,
                          structure_guidance=None, inference_size=None, lr = 5e-2, opt_steps=100,**kwargs):
        """
        ODE solving from z0 (ims) to z1 (depth).
        """
        ode_kwargs = dict(method="euler", rtol=1e-5, atol=1e-5, options=dict(step_size=1.0 / num_steps))
        
        # t specifies which intermediate times should the solver return
        # e.g. t = [0, 0.5, 1] means return the solution at t=0, t=0.5 and t=1
        # but it also specifies the number of steps for fixed step size methods
        t = torch.linspace(0, 1, n_intermediates + 2, device=z.device, dtype=z.dtype)
        # t = torch.tensor([0., 1.], device=z.device, dtype=z.dtype)

        # allow conditioning information for model
        ode_fn = partial(self.ode_fn, **kwargs)
        
        ode_results = odeint(ode_fn, z, t, **ode_kwargs)
        
        # SECTION - Alignment
        measurement_loss = torch.nn.L1Loss()
        
        depth_latent_pseudo_z0 = ode_results[-1].detach().clone()
        depth_latent_pseudo_z0 = depth_latent_pseudo_z0.requires_grad_(True)
        
        optimizer = torch.optim.AdamW([depth_latent_pseudo_z0], lr=lr)                 
        
        for _ in tqdm(range(opt_steps)):
            optimizer.zero_grad()
            depth_pseudo_x0 = self.decode(depth_latent_pseudo_z0)
            depth_pseudo_x0 = depth_pseudo_x0.mean(dim=1, keepdim=True)
            
            if inference_size is not None:
                depth_pseudo_x0 = F.interpolate(depth_pseudo_x0, size=sparse_depth_map.squeeze().shape, mode='bilinear', align_corners=True, antialias=True)#ANCHOR - .squeeze()
        
            depth_pseudo_x0 = per_sample_min_max_normalization(depth_pseudo_x0.exp())
        
            mask = norm_sparse_depth > 0
        
            l_measurement = measurement_loss(norm_sparse_depth[mask], depth_pseudo_x0[mask] )
            l_local_smoothness = self.local_smoothness_loss(structure_guidance,depth_pseudo_x0) *0.1 #0.1
            if structure_guidance.shape[1] == 3: 
                l_ssim = (1- ssimloss(depth_pseudo_x0, structure_guidance[:,1,...])) *0.1
            else:
                l_ssim = (1- ssimloss(depth_pseudo_x0, structure_guidance)) *0.1
            
            loss = l_measurement + l_local_smoothness + l_ssim
            
            loss.backward() # Take GD step
            optimizer.step()
                        
            with torch.no_grad():
                self.metric.evaluate(depth_pseudo_x0.squeeze().detach().cpu().numpy(), gt_depth_map.squeeze(), sparse_depth_map )
        
        
        # if n_intermediates > 0:
        #     return ode_results
        # return ode_results[-1]
        return depth_latent_pseudo_z0   
    
    def generate_alignment_multi_step(self, z: Tensor, num_steps: int = 4, n_intermediates: int = 0, img=None, sparse_depth_map=None, norm_sparse_depth=None, gt_depth_map=None,
                            structure_guidance=None, inference_size=None, lr = 5e-2, opt_steps=100, **kwargs):
        """
        ODE solving from z0 (ims) to z1 (depth).
        """
        ode_kwargs = dict(method="euler", rtol=1e-5, atol=1e-5, options=dict(step_size=1.0 / num_steps))
        
        # t specifies which intermediate times should the solver return
        # e.g. t = [0, 0.5, 1] means return the solution at t=0, t=0.5 and t=1
        # but it also specifies the number of steps for fixed step size methods
        
        # t = torch.tensor([0., 1.], device=z.device, dtype=z.dtype)

        # allow conditioning information for model
        ode_fn = partial(self.ode_fn, **kwargs)
        
        # SECTION - Alignment
        measurement_loss = torch.nn.L1Loss()
        
        current_z = z.detach()
        
        # Number of sampling steps
        for i in range(n_intermediates+1):
            t = torch.linspace(0, 1, i + 2, device=z.device, dtype=z.dtype)[-2:]
            ode_results = odeint(ode_fn, current_z, t, **ode_kwargs)        
        
            depth_latent_pseudo_x0 = ode_results[-1].detach().clone()
            depth_latent_pseudo_x0 = depth_latent_pseudo_x0.requires_grad_(True)
        
            optimizer = torch.optim.AdamW([depth_latent_pseudo_x0], lr=lr) 
            
            # Aligning each intermediate latent with measurement
            for _ in tqdm(range(opt_steps)):
                optimizer.zero_grad()
                depth_pseudo_x0 = self.decode(depth_latent_pseudo_x0)
                depth_pseudo_x0 = depth_pseudo_x0.mean(dim=1, keepdim=True)
                
                if inference_size is not None:
                    depth_pseudo_x0 = F.interpolate(depth_pseudo_x0, size=sparse_depth_map.squeeze().shape, mode='bilinear', align_corners=True, antialias=True)#ANCHOR - .squeeze()
        
                depth_pseudo_x0 = per_sample_min_max_normalization(depth_pseudo_x0.exp())
            
                mask = norm_sparse_depth > 0
            
                l_measurement = measurement_loss(norm_sparse_depth[mask], depth_pseudo_x0[mask] )
                l_local_smoothness = self.local_smoothness_loss(structure_guidance,depth_pseudo_x0) *0.1 #0.1
                if structure_guidance.shape[1] == 3: 
                    l_ssim = (1- ssimloss(depth_pseudo_x0, structure_guidance[:,1,...])) *0.1
                else:
                    l_ssim = (1- ssimloss(depth_pseudo_x0, structure_guidance)) *0.1
                
                loss = l_measurement + l_local_smoothness + l_ssim
                
                loss.backward() # Take GD step
                optimizer.step()
                
                with torch.no_grad():
                    self.metric.evaluate(depth_pseudo_x0.squeeze().detach().cpu().numpy(), gt_depth_map.squeeze(), sparse_depth_map )

            current_z = depth_latent_pseudo_x0.detach()
            
            # Resample
            if i != n_intermediates:
                current_z = q_sample(current_z, self.noising_step//(i+2))    
            
        return depth_latent_pseudo_x0      
        
    
    def forward(self, ims: Tensor, num_steps: int = 4, ensemble_size: int = 1, norm_sparse_depth=None, sparse_depth_map=None, gt_depth_map=None, n_intermediates=0, relative_structure_depth=None, inference_size=None):
        """
        Args:
            ims: Tensor of shape (b, 3, h, w) in range [-1, 1]
        Returns:
            depth: Tensor of shape (b, 1, h, w) in range [0, 1]
        """
        if ensemble_size > 1:
            assert ims.shape[0] == 1, "Ensemble mode only supported with batch size 1"
            ims = ims.repeat(ensemble_size, 1, 1, 1)
        
        bs, dev = ims.shape[0], ims.device

        ims_z = self.encode(ims, sample_posterior=False)

        conditioning = torch.tensor(self.empty_text_embed).to(dev).repeat(bs, 1, 1)
        context = ims_z
        
        x_source = ims_z

        if self.noising_step > 0:
            x_source = q_sample(x_source, self.noising_step)    
            
        # For Structure Similarity Guidance
        if relative_structure_depth is not None:
                        
            # If relative structure depth shape is different
            if relative_structure_depth.shape[2] != norm_sparse_depth.shape[2] or relative_structure_depth.shape[3] != norm_sparse_depth.shape[3]:
                target_size = (norm_sparse_depth.shape[2] , norm_sparse_depth.shape[3] )
                relative_structure_depth = F.interpolate(relative_structure_depth, size=target_size, mode='bilinear', align_corners=True, antialias=True)
        
        if relative_structure_depth is None: 
            structure_rgb_in = (ims + 1.0) / 2.0
            if inference_size is not None:
                structure_rgb_in = F.interpolate(structure_rgb_in, size=sparse_depth_map.squeeze().shape, mode='bilinear', align_corners=True, antialias=True)#.squeeze()            

        if relative_structure_depth is not None: structure_guidance = relative_structure_depth
        else: structure_guidance = structure_rgb_in

        # Alignned depth generation
        if sparse_depth_map is not None:
            
            if n_intermediates >0:
                depth_z = self.generate_alignment_multi_step(x_source, num_steps=num_steps, context=context, context_ca=conditioning,
                                                             img= ims, sparse_depth_map=sparse_depth_map, norm_sparse_depth=norm_sparse_depth, gt_depth_map=gt_depth_map, n_intermediates=n_intermediates,
                                                             structure_guidance=structure_guidance, inference_size=inference_size)
            else:
                depth_z = self.generate_alignment_one_step(x_source, num_steps=num_steps, context=context, context_ca=conditioning,
                                                           img= ims, sparse_depth_map=sparse_depth_map, norm_sparse_depth=norm_sparse_depth, gt_depth_map=gt_depth_map,
                                                           structure_guidance=structure_guidance, inference_size=inference_size)
            
        # Original forward
        else:
            depth_z = self.generate(x_source, num_steps=num_steps, context=context, context_ca=conditioning)


        depth = self.decode(depth_z)
        depth = depth.mean(dim=1, keepdim=True)

        if ensemble_size > 1:
            depth = depth.mean(dim=0, keepdim=True)
        
        # normalize depth maps to range [-1, 1]
        depth = per_sample_min_max_normalization(depth.exp())

        return depth
    
    def local_smoothness_loss(self, image, depth_map):
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
    
    @torch.no_grad()
    def predict_depth(self, ims: Tensor, num_steps: int = 4, ensemble_size: int = 1):
        """ Inference method for DepthFM. """
        return self.forward(ims, num_steps, ensemble_size)
    
    @torch.no_grad()
    def encode(self, x: Tensor, sample_posterior: bool = True):
        posterior = self.vae.encode(x)
        if sample_posterior:
            z = posterior.latent_dist.sample()
        else:
            z = posterior.latent_dist.mode()
        # normalize latent code
        z = z * self.scale_factor
        return z
    
    def decode(self, z: Tensor):
        # with torch.no_grad():
        z = 1.0 / self.scale_factor * z
        return self.vae.decode(z).sample


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def cosine_log_snr(t, eps=0.00001):
    """
    Returns log Signal-to-Noise ratio for time step t and image size 64
    eps: avoid division by zero
    """
    return -2 * np.log(np.tan((np.pi * t) / 2) + eps)


def cosine_alpha_bar(t):
    return sigmoid(cosine_log_snr(t))


def q_sample(x_start: torch.Tensor, t: int, noise: torch.Tensor = None, n_diffusion_timesteps: int = 1000):
    """
    Diffuse the data for a given number of diffusion steps. In other
    words sample from q(x_t | x_0).
    """
    dev = x_start.device
    dtype = x_start.dtype

    if noise is None:
        noise = torch.randn_like(x_start)
    
    alpha_bar_t = cosine_alpha_bar(t / n_diffusion_timesteps)
    alpha_bar_t = torch.tensor(alpha_bar_t).to(dev).to(dtype)

    return torch.sqrt(alpha_bar_t) * x_start + torch.sqrt(1 - alpha_bar_t) * noise


def per_sample_min_max_normalization(x):
    """ Normalize each sample in a batch independently
    with min-max normalization to [0, 1] """
    bs, *shape = x.shape
    x_ = einops.rearrange(x, "b ... -> b (...)")
    min_val = einops.reduce(x_, "b ... -> b", "min")[..., None]
    max_val = einops.reduce(x_, "b ... -> b", "max")[..., None]
    x_ = (x_ - min_val) / (max_val - min_val)
    return x_.reshape(bs, *shape)
