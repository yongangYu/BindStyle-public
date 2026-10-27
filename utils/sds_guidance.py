import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class NullSDSGuidance(nn.Module):
    enabled = False

    def forward(self, image):
        return image.sum() * 0.0


class DeepFloydSDSGuidance(nn.Module):
    enabled = True

    def __init__(self, cfg, device):
        super().__init__()
        try:
            from diffusers import DDPMScheduler, IFPipeline
        except (ImportError, RuntimeError) as exc:
            if "cached_download" in str(exc) and "huggingface_hub" in str(exc):
                raise ImportError(
                    "diffusers and huggingface_hub are version-incompatible: "
                    "the installed diffusers imports huggingface_hub.cached_download, "
                    "which is missing in the installed huggingface_hub. Fix the "
                    "environment by upgrading diffusers or downgrading "
                    "huggingface_hub, then enable SDS again."
                ) from exc
            if "Dinov2WithRegistersConfig" in str(exc):
                raise ImportError(
                    "diffusers and transformers are version-incompatible: "
                    "the installed diffusers imports Dinov2WithRegistersConfig, "
                    "which is missing in the installed transformers. Upgrade "
                    "transformers, then enable SDS again."
                ) from exc
            raise ImportError(
                "diffusers is required for DeepFloyd SDS. Install diffusers and "
                "enable optim.mesh_style_update.sds.enabled=true again."
            ) from exc

        self.device = torch.device(device)
        self.model_name = cfg.get("model_name", "DeepFloyd/IF-I-M-v1.0")
        self.cache_dir = cfg.get("cache_dir", None) or None
        self.revision = cfg.get("revision", None) or None
        self.variant = cfg.get("variant", None) or None
        self.local_files_only = bool(cfg.get("local_files_only", False))
        self.token = cfg.get("token", None) or None
        self.prompt = cfg.get("prompt", "")
        if not self.prompt:
            raise ValueError(
                "optim.mesh_style_update.sds.prompt is required for deepfloyd_if SDS."
            )
        self.negative_prompt = cfg.get("negative_prompt", "")
        self.guidance_scale = float(cfg.get("guidance_scale", 50.0))
        self.image_size = int(cfg.get("image_size", 64))
        self.t_min = float(cfg.get("t_min", 0.02))
        self.t_max = float(cfg.get("t_max", 0.98))
        self.grad_clip = float(cfg.get("sds_grad_clip", 0.0))

        torch_dtype = torch.float16 if bool(cfg.get("half_precision", True)) else torch.float32
        self.pipe = IFPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            revision=self.revision,
            token=self.token,
            safety_checker=None,
            watermarker=None,
            variant=self.variant,
        )
        self.pipe.to(self.device)
        self.pipe.unet.eval()
        self.pipe.text_encoder.eval()
        for param in self.pipe.unet.parameters():
            param.requires_grad_(False)
        for param in self.pipe.text_encoder.parameters():
            param.requires_grad_(False)

        self.scheduler = DDPMScheduler.from_config(self.pipe.scheduler.config)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        self.num_train_timesteps = int(self.scheduler.config.num_train_timesteps)
        self.min_step = max(0, int(self.num_train_timesteps * self.t_min))
        self.max_step = min(
            self.num_train_timesteps - 1,
            int(self.num_train_timesteps * self.t_max),
        )
        self.register_buffer(
            "alphas_cumprod",
            self.scheduler.alphas_cumprod.detach().float(),
        )
        self._encode_prompt()

    def _encode_prompt(self):
        with torch.no_grad():
            try:
                prompt_embeds, negative_embeds = self.pipe.encode_prompt(
                    prompt=self.prompt,
                    device=self.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=True,
                    negative_prompt=self.negative_prompt,
                )
            except TypeError:
                encoded = self.pipe.encode_prompt(
                    self.prompt,
                    self.device,
                    1,
                    True,
                    self.negative_prompt,
                )
                prompt_embeds, negative_embeds = encoded[:2]
        self.prompt_embeds = prompt_embeds.detach()
        self.negative_prompt_embeds = negative_embeds.detach()

    def forward(self, image):
        image = F.interpolate(
            image,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        image_model = image * 2.0 - 1.0
        image_model = image_model.to(
            device=self.device,
            dtype=self.prompt_embeds.dtype,
        )

        batch_size = image_model.shape[0]
        timesteps = torch.randint(
            self.min_step,
            self.max_step + 1,
            (batch_size,),
            device=self.device,
            dtype=torch.long,
        )
        noise = torch.randn_like(image_model)
        noisy = self.scheduler.add_noise(image_model, noise, timesteps)

        model_input = torch.cat([noisy, noisy], dim=0)
        timestep_input = torch.cat([timesteps, timesteps], dim=0)
        prompt_embeds = self.prompt_embeds.expand(batch_size, -1, -1)
        negative_embeds = self.negative_prompt_embeds.expand(batch_size, -1, -1)
        encoder_hidden_states = torch.cat([negative_embeds, prompt_embeds], dim=0)

        with torch.no_grad():
            noise_pred = self.pipe.unet(
                model_input,
                timestep_input,
                encoder_hidden_states=encoder_hidden_states,
            ).sample
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + self.guidance_scale * (
                noise_text - noise_uncond
            )

        weight = (1.0 - self.alphas_cumprod[timesteps]).view(batch_size, 1, 1, 1)
        grad = weight.to(noise_pred.dtype) * (noise_pred - noise)
        if self.grad_clip > 0:
            grad = grad.clamp(-self.grad_clip, self.grad_clip)
        grad = torch.nan_to_num(grad)

        return (grad.detach() * image_model).sum() / max(batch_size, 1)


class ImageVariationSDSGuidance(nn.Module):
    enabled = True

    def __init__(self, cfg, device):
        super().__init__()
        try:
            from diffusers import DDPMScheduler, StableDiffusionImageVariationPipeline
        except (ImportError, RuntimeError) as exc:
            if "cached_download" in str(exc) and "huggingface_hub" in str(exc):
                raise ImportError(
                    "diffusers and huggingface_hub are version-incompatible: "
                    "the installed diffusers imports huggingface_hub.cached_download, "
                    "which is missing in the installed huggingface_hub. Fix the "
                    "environment by upgrading diffusers or downgrading "
                    "huggingface_hub, then enable image-variation SDS again."
                ) from exc
            if "Dinov2WithRegistersConfig" in str(exc):
                raise ImportError(
                    "diffusers and transformers are version-incompatible: "
                    "the installed diffusers imports Dinov2WithRegistersConfig, "
                    "which is missing in the installed transformers. Upgrade "
                    "transformers, then enable image-variation SDS again."
                ) from exc
            raise ImportError(
                "diffusers with StableDiffusionImageVariationPipeline is required "
                "for image-variation SDS."
            ) from exc

        self.device = torch.device(device)
        self.model_name = cfg.get(
            "model_name",
            "lambda/sd-image-variations-diffusers",
        )
        self.cache_dir = cfg.get("cache_dir", None) or None
        self.revision = cfg.get("revision", None) or None
        self.variant = cfg.get("variant", None) or None
        self.local_files_only = bool(cfg.get("local_files_only", False))
        self.token = cfg.get("token", None) or None
        self.style_path = cfg.get("style_path", "")
        if not self.style_path:
            raise ValueError(
                "optim.mesh_style_update.sds.style_path or "
                "optim.mesh_style_update.style_path is required for "
                "image-variation SDS."
            )

        self.guidance_scale = float(cfg.get("guidance_scale", 3.0))
        self.image_size = int(cfg.get("image_size", 512))
        self.t_min = float(cfg.get("t_min", 0.02))
        self.t_max = float(cfg.get("t_max", 0.98))
        self.grad_clip = float(cfg.get("sds_grad_clip", 0.0))
        self.vae_latent_mode = str(cfg.get("vae_latent_mode", "mode")).lower()

        torch_dtype = torch.float16 if bool(cfg.get("half_precision", True)) else torch.float32
        self.pipe = StableDiffusionImageVariationPipeline.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
            revision=self.revision,
            token=self.token,
            safety_checker=None,
            requires_safety_checker=False,
            variant=self.variant,
        )
        self.pipe.to(self.device)
        self.pipe.vae.eval()
        self.pipe.unet.eval()
        self.pipe.image_encoder.eval()
        for module in [self.pipe.vae, self.pipe.unet, self.pipe.image_encoder]:
            for param in module.parameters():
                param.requires_grad_(False)

        self.scheduler = DDPMScheduler.from_config(self.pipe.scheduler.config)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        self.num_train_timesteps = int(self.scheduler.config.num_train_timesteps)
        self.min_step = max(0, int(self.num_train_timesteps * self.t_min))
        self.max_step = min(
            self.num_train_timesteps - 1,
            int(self.num_train_timesteps * self.t_max),
        )
        if self.max_step <= self.min_step:
            raise ValueError(
                "Invalid SDS timestep range: t_max must produce a larger step "
                "than t_min."
            )
        self.register_buffer(
            "alphas_cumprod",
            self.scheduler.alphas_cumprod.detach().float(),
        )
        self._encode_style_image()

    def _encode_style_image(self):
        style_image = Image.open(self.style_path).convert("RGB")
        with torch.no_grad():
            image_embeds = self.pipe._encode_image(
                style_image,
                self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
        if image_embeds.dim() == 2:
            image_embeds = image_embeds.unsqueeze(1)
        self.style_image_embeds = image_embeds.detach()
        self.uncond_image_embeds = torch.zeros_like(self.style_image_embeds)

    def _encode_latents(self, image):
        image = F.interpolate(
            image,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        image_model = image * 2.0 - 1.0
        vae_dtype = next(self.pipe.vae.parameters()).dtype
        image_model = image_model.to(device=self.device, dtype=vae_dtype)
        latent_dist = self.pipe.vae.encode(image_model).latent_dist
        if self.vae_latent_mode == "sample":
            latents = latent_dist.sample()
        else:
            try:
                latents = latent_dist.mode()
            except AttributeError:
                latents = latent_dist.mean
        scaling_factor = getattr(self.pipe.vae.config, "scaling_factor", 0.18215)
        return latents * scaling_factor

    def _prediction_to_epsilon(self, prediction, noisy_latents, timesteps):
        if hasattr(self.scheduler.config, "get"):
            prediction_type = self.scheduler.config.get("prediction_type", "epsilon")
        else:
            prediction_type = getattr(self.scheduler.config, "prediction_type", "epsilon")
        if prediction_type == "epsilon":
            return prediction
        if prediction_type == "v_prediction":
            alpha_prod_t = self.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
            beta_prod_t = 1.0 - alpha_prod_t
            return alpha_prod_t.sqrt() * prediction + beta_prod_t.sqrt() * noisy_latents
        return prediction

    def forward(self, image):
        latents = self._encode_latents(image)
        unet_dtype = next(self.pipe.unet.parameters()).dtype
        latents = latents.to(dtype=unet_dtype)

        batch_size = latents.shape[0]
        timesteps = torch.randint(
            self.min_step,
            self.max_step + 1,
            (batch_size,),
            device=self.device,
            dtype=torch.long,
        )
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        style_embeds = self.style_image_embeds.to(dtype=unet_dtype).expand(
            batch_size,
            -1,
            -1,
        )
        do_cfg = self.guidance_scale > 1.0
        if do_cfg:
            uncond_embeds = self.uncond_image_embeds.to(dtype=unet_dtype).expand(
                batch_size,
                -1,
                -1,
            )
            encoder_hidden_states = torch.cat([uncond_embeds, style_embeds], dim=0)
            model_input = torch.cat([noisy_latents, noisy_latents], dim=0)
            timestep_input = torch.cat([timesteps, timesteps], dim=0)
        else:
            encoder_hidden_states = style_embeds
            model_input = noisy_latents
            timestep_input = timesteps

        model_input = self.scheduler.scale_model_input(model_input, timestep_input)
        with torch.no_grad():
            noise_pred = self.pipe.unet(
                model_input,
                timestep_input,
                encoder_hidden_states=encoder_hidden_states,
            ).sample
            if do_cfg:
                noise_uncond, noise_cond = noise_pred.chunk(2)
                noise_pred = noise_uncond + self.guidance_scale * (
                    noise_cond - noise_uncond
                )
            noise_pred = self._prediction_to_epsilon(
                noise_pred,
                noisy_latents,
                timesteps,
            )

        weight = (1.0 - self.alphas_cumprod[timesteps]).view(batch_size, 1, 1, 1)
        grad = weight.to(noise_pred.dtype) * (noise_pred - noise)
        if self.grad_clip > 0:
            grad = grad.clamp(-self.grad_clip, self.grad_clip)
        grad = torch.nan_to_num(grad)

        return (grad.detach() * latents).sum() / max(batch_size, 1)


class ImageConditionedSDSGuidance(ImageVariationSDSGuidance):
    pass


def build_sds_guidance(cfg, device):
    sds_cfg = cfg.get("sds", {})
    if not sds_cfg or not sds_cfg.get("enabled", False):
        return NullSDSGuidance()

    provider = str(sds_cfg.get("provider", "deepfloyd_if")).lower()
    if provider == "deepfloyd_if":
        return DeepFloydSDSGuidance(sds_cfg, device)
    if provider in ["image", "image_conditioned", "image_variation", "sd_image_variation"]:
        if (not sds_cfg.get("style_path", "")) and cfg.get("style_path", ""):
            sds_cfg = dict(sds_cfg)
            sds_cfg["style_path"] = cfg.get("style_path", "")
        return ImageVariationSDSGuidance(sds_cfg, device)
    if provider in ["sdxl_ip_adapter", "sdxl_ip", "ip_adapter_sdxl"]:
        if (not sds_cfg.get("style_path", "")) and cfg.get("style_path", ""):
            sds_cfg = dict(sds_cfg)
            sds_cfg["style_path"] = cfg.get("style_path", "")
        from utils.sdxl_ip_adapter_sds_guidance import SDXLIPAdapterSDSGuidance

        return SDXLIPAdapterSDSGuidance(sds_cfg, device)
    if provider in [
        "sdxl_animatediff_controlnet",
        "sdxl_video_controlnet",
        "animatediff_sdxl_controlnet",
    ]:
        if (not sds_cfg.get("style_path", "")) and cfg.get("style_path", ""):
            sds_cfg = dict(sds_cfg)
            sds_cfg["style_path"] = cfg.get("style_path", "")
        from utils.sdxl_animatediff_controlnet_csd_guidance import (
            SDXLAnimateDiffControlNetCSDGuidance,
        )

        return SDXLAnimateDiffControlNetCSDGuidance(sds_cfg, device)
    if provider == "ip_adapter":
        raise ValueError(
            "Use provider=sdxl_ip_adapter for SDXL/IP-Adapter guidance, or "
            "provider=image_variation for legacy image-variation SDS."
        )
    raise ValueError(f"Unsupported SDS provider: {provider}")
