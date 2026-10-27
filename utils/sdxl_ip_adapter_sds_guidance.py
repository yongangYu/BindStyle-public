import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class SDXLIPAdapterSDSGuidance(nn.Module):
    """Frozen SDXL/IP-Adapter score model with a differentiable VAE input."""

    enabled = True

    def __init__(self, cfg, device):
        super().__init__()
        try:
            from diffusers import (
                DDIMScheduler,
                DDPMScheduler,
                StableDiffusionXLPipeline,
            )
            from transformers import CLIPVisionModelWithProjection
        except (ImportError, RuntimeError) as exc:
            raise ImportError(
                "diffusers with StableDiffusionXLPipeline/IP-Adapter support and "
                "transformers with CLIPVisionModelWithProjection are required for "
                "sdxl_ip_adapter SDS."
            ) from exc

        self.device = torch.device(device)
        self.model_name = cfg.get(
            "model_name", "stabilityai/stable-diffusion-xl-base-1.0"
        )
        self.ip_adapter_name = cfg.get("ip_adapter_name", "h94/IP-Adapter")
        self.ip_adapter_subfolder = cfg.get("ip_adapter_subfolder", "sdxl_models")
        self.ip_adapter_weight_name = cfg.get(
            "ip_adapter_weight_name", "ip-adapter_sdxl.bin"
        )
        self.image_encoder_folder = cfg.get(
            "image_encoder_folder", "image_encoder"
        )
        self.cache_dir = cfg.get("cache_dir", None) or None
        self.local_files_only = bool(cfg.get("local_files_only", False))
        self.revision = cfg.get("revision", None) or None
        self.variant = cfg.get("variant", None) or None
        self.token = cfg.get("token", None) or None
        self.style_path = cfg.get("style_path", "")
        if not self.style_path:
            raise ValueError(
                "optim.gaussian_style_update.style_path or its sds.style_path is "
                "required for sdxl_ip_adapter SDS."
            )

        self.prompt = cfg.get("prompt", "")
        self.negative_prompt = cfg.get("negative_prompt", "")
        self.gradient_mode = str(cfg.get("gradient_mode", "csd")).lower()
        if self.gradient_mode not in ["csd", "sds"]:
            raise ValueError("SDXL guidance gradient_mode must be 'csd' or 'sds'.")
        self.guidance_scale = float(cfg.get("guidance_scale", 5.0))
        self.ip_adapter_scale = float(cfg.get("ip_adapter_scale", 0.7))
        self.image_size = int(cfg.get("image_size", 512))
        if self.image_size <= 0 or self.image_size % 8 != 0:
            raise ValueError("SDXL SDS image_size must be a positive multiple of 8.")
        self.t_min = float(cfg.get("t_min", 0.05))
        self.t_max = float(cfg.get("t_max", 0.85))
        self.grad_clip = float(cfg.get("sds_grad_clip", 0.0))
        self.vae_latent_mode = str(cfg.get("vae_latent_mode", "mode")).lower()
        self.offload_condition_encoders = bool(
            cfg.get("offload_condition_encoders", True)
        )

        torch_dtype = (
            torch.float16 if bool(cfg.get("half_precision", True)) else torch.float32
        )
        load_kwargs = {
            "torch_dtype": torch_dtype,
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only,
        }
        if self.revision:
            load_kwargs["revision"] = self.revision
        if self.variant:
            load_kwargs["variant"] = self.variant
        if self.token:
            load_kwargs["token"] = self.token

        self._validate_local_model_paths()
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            self.model_name,
            **load_kwargs,
        )
        ip_kwargs = {
            "weight_name": self.ip_adapter_weight_name,
            "local_files_only": self.local_files_only,
        }
        if self.ip_adapter_subfolder:
            ip_kwargs["subfolder"] = self.ip_adapter_subfolder
        local_image_encoder_dir = self._local_image_encoder_dir()
        if local_image_encoder_dir is not None:
            image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                local_image_encoder_dir,
                torch_dtype=torch_dtype,
                local_files_only=True,
            )
            self.pipe.register_modules(image_encoder=image_encoder)
            ip_kwargs["image_encoder_folder"] = None
        elif self.image_encoder_folder:
            ip_kwargs["image_encoder_folder"] = self.image_encoder_folder
        if self.cache_dir:
            ip_kwargs["cache_dir"] = self.cache_dir
        if self.token:
            ip_kwargs["token"] = self.token
        self.pipe.load_ip_adapter(self.ip_adapter_name, **ip_kwargs)
        self.pipe.set_ip_adapter_scale(self.ip_adapter_scale)
        self.pipe.to(self.device)

        # The stock SDXL VAE is safer in float32 during optimization.
        if not bool(cfg.get("vae_half_precision", False)):
            self.pipe.vae.to(dtype=torch.float32)

        for module_name in [
            "vae",
            "unet",
            "text_encoder",
            "text_encoder_2",
            "image_encoder",
        ]:
            module = getattr(self.pipe, module_name, None)
            if module is None:
                continue
            module.eval()
            for param in module.parameters():
                param.requires_grad_(False)

        self.scheduler = DDPMScheduler.from_config(self.pipe.scheduler.config)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        self.trajectory_scheduler = DDIMScheduler.from_config(
            self.pipe.scheduler.config,
            clip_sample=False,
            set_alpha_to_one=False,
        )
        self.trajectory_scheduler.alphas_cumprod = (
            self.trajectory_scheduler.alphas_cumprod.to(self.device)
        )
        self.num_train_timesteps = int(self.scheduler.config.num_train_timesteps)
        self.min_step = max(0, int(self.num_train_timesteps * self.t_min))
        self.max_step = min(
            self.num_train_timesteps - 1,
            int(self.num_train_timesteps * self.t_max),
        )
        if self.max_step <= self.min_step:
            raise ValueError(
                "Invalid SDXL SDS timestep range: t_max must be greater than t_min."
            )
        self.register_buffer(
            "alphas_cumprod",
            self.scheduler.alphas_cumprod.detach().float(),
        )

        self.source_cache = {}
        self._encode_conditions()
        if self.offload_condition_encoders:
            for module_name in ["text_encoder", "text_encoder_2", "image_encoder"]:
                module = getattr(self.pipe, module_name, None)
                if module is not None:
                    module.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _local_image_encoder_dir(self):
        if not self.image_encoder_folder or not os.path.isdir(self.ip_adapter_name):
            return None
        adapter_dir = self.ip_adapter_name
        if self.ip_adapter_subfolder:
            adapter_dir = os.path.join(adapter_dir, self.ip_adapter_subfolder)
        return os.path.join(adapter_dir, self.image_encoder_folder)

    def _validate_local_model_paths(self):
        if os.path.isabs(self.model_name):
            if not os.path.isdir(self.model_name):
                raise FileNotFoundError(
                    f"Local SDXL directory does not exist: {self.model_name}"
                )
            model_index = os.path.join(self.model_name, "model_index.json")
            if not os.path.isfile(model_index):
                raise FileNotFoundError(
                    "The local SDXL directory is incomplete; model_index.json "
                    f"was not found at: {model_index}"
                )

        if not os.path.isabs(self.ip_adapter_name):
            return
        if not os.path.isdir(self.ip_adapter_name):
            raise FileNotFoundError(
                f"Local IP-Adapter directory does not exist: {self.ip_adapter_name}"
            )
        adapter_dir = self.ip_adapter_name
        if self.ip_adapter_subfolder:
            adapter_dir = os.path.join(adapter_dir, self.ip_adapter_subfolder)
        adapter_weight = os.path.join(adapter_dir, self.ip_adapter_weight_name)
        if not os.path.isfile(adapter_weight):
            raise FileNotFoundError(
                f"Local IP-Adapter weight was not found: {adapter_weight}"
            )
        image_encoder_dir = self._local_image_encoder_dir()
        if image_encoder_dir is not None:
            if not os.path.isdir(image_encoder_dir):
                raise FileNotFoundError(
                    "Local IP-Adapter image encoder directory was not found: "
                    f"{image_encoder_dir}"
                )
            image_encoder_config = os.path.join(image_encoder_dir, "config.json")
            if not os.path.isfile(image_encoder_config):
                raise FileNotFoundError(
                    "Local IP-Adapter image encoder config was not found: "
                    f"{image_encoder_config}"
                )
            image_encoder_weights = [
                os.path.join(image_encoder_dir, "model.safetensors"),
                os.path.join(image_encoder_dir, "pytorch_model.bin"),
            ]
            if not any(os.path.isfile(path) for path in image_encoder_weights):
                raise FileNotFoundError(
                    "Local IP-Adapter image encoder weights were not found in: "
                    f"{image_encoder_dir}"
                )

    @torch.no_grad()
    def _encode_conditions(self):
        encoded = self.pipe.encode_prompt(
            prompt=self.prompt,
            prompt_2=self.prompt,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=self.negative_prompt,
            negative_prompt_2=self.negative_prompt,
        )
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = encoded[:4]
        self.prompt_embeds = prompt_embeds.detach()
        self.negative_prompt_embeds = negative_prompt_embeds.detach()
        self.pooled_prompt_embeds = pooled_prompt_embeds.detach()
        self.negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.detach()

        style_image = Image.open(self.style_path).convert("RGB")
        image_embeds = self.pipe.prepare_ip_adapter_image_embeds(
            ip_adapter_image=style_image,
            ip_adapter_image_embeds=None,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )
        self.cond_image_embeds = []
        self.uncond_image_embeds = []
        for embeds in image_embeds:
            uncond, cond = embeds.chunk(2, dim=0)
            self.uncond_image_embeds.append(uncond.detach())
            self.cond_image_embeds.append(cond.detach())

        projection_dim = self.pooled_prompt_embeds.shape[-1]
        self.add_time_ids = self.pipe._get_add_time_ids(
            (self.image_size, self.image_size),
            (0, 0),
            (self.image_size, self.image_size),
            dtype=self.prompt_embeds.dtype,
            text_encoder_projection_dim=projection_dim,
        ).detach().to(self.device)

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
        return latents * self.pipe.vae.config.scaling_factor

    @torch.no_grad()
    def decode_latents(self, latents):
        vae_dtype = next(self.pipe.vae.parameters()).dtype
        latents = latents.to(device=self.device, dtype=vae_dtype)
        latents = latents / self.pipe.vae.config.scaling_factor
        image = self.pipe.vae.decode(latents).sample
        return (image * 0.5 + 0.5).clamp(0.0, 1.0).float()

    @staticmethod
    def _image_to_batch(image):
        if isinstance(image, Image.Image):
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            image = torch.from_numpy(array).permute(2, 0, 1)
        if not torch.is_tensor(image):
            raise TypeError("Source images must be PIL images or torch tensors.")
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.dim() != 4 or image.shape[1] != 3:
            raise ValueError("Source image tensors must have shape [B, 3, H, W].")
        image = image.detach().float()
        if image.max() > 1.0:
            image = image / 255.0
        return image.clamp(0.0, 1.0)

    @staticmethod
    def _batch_to_pil(image):
        image = image[0].detach().float().cpu().clamp(0.0, 1.0)
        array = image.permute(1, 2, 0).mul(255.0).round().byte().numpy()
        return Image.fromarray(array, mode="RGB")

    @torch.no_grad()
    def cache_source_images(self, source_images):
        """Cache content latents and image conditions for CSD/trajectory views."""
        if self.gradient_mode != "csd":
            return
        if not isinstance(source_images, dict):
            source_images = dict(source_images)
        image_encoder = getattr(self.pipe, "image_encoder", None)
        if image_encoder is None:
            raise RuntimeError("CSD requires the IP-Adapter image encoder.")
        image_encoder.to(self.device)

        for key, source_image in source_images.items():
            source_batch = self._image_to_batch(source_image).to(self.device)
            source_latents = self._encode_latents(source_batch).detach().cpu()
            source_pil = self._batch_to_pil(source_batch)
            image_embeds = self.pipe.prepare_ip_adapter_image_embeds(
                ip_adapter_image=source_pil,
                ip_adapter_image_embeds=None,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
            )
            source_uncond = []
            source_cond = []
            for embeds in image_embeds:
                uncond, cond = embeds.chunk(2, dim=0)
                source_uncond.append(uncond.detach().cpu())
                source_cond.append(cond.detach().cpu())
            self.source_cache[key] = {
                "latents": source_latents,
                "uncond_image_embeds": source_uncond,
                "cond_image_embeds": source_cond,
            }

        if self.offload_condition_encoders:
            image_encoder.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(
            f"[MaGS] Cached {len(self.source_cache)} content image conditions "
            "for CSD/trajectory guidance."
        )

    def _prediction_to_epsilon(self, prediction, noisy_latents, timesteps):
        if hasattr(self.scheduler.config, "get"):
            prediction_type = self.scheduler.config.get("prediction_type", "epsilon")
        else:
            prediction_type = getattr(
                self.scheduler.config, "prediction_type", "epsilon"
            )
        if prediction_type == "epsilon":
            return prediction
        if prediction_type == "v_prediction":
            alpha_prod_t = self.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
            beta_prod_t = 1.0 - alpha_prod_t
            return alpha_prod_t.sqrt() * prediction + beta_prod_t.sqrt() * noisy_latents
        return prediction

    @staticmethod
    def _repeat_condition(condition, batch_size):
        return condition.expand(batch_size, *condition.shape[1:])

    def _trajectory_condition_batch(self, source_pkg, batch_size, unet_dtype):
        prompt_embeds = self._repeat_condition(self.prompt_embeds, batch_size)
        negative_prompt_embeds = self._repeat_condition(
            self.negative_prompt_embeds, batch_size
        )
        pooled_embeds = self._repeat_condition(
            self.pooled_prompt_embeds, batch_size
        )
        negative_pooled_embeds = self._repeat_condition(
            self.negative_pooled_prompt_embeds, batch_size
        )
        add_time_ids = self._repeat_condition(self.add_time_ids, batch_size)

        source_uncond_embeds = [
            self._repeat_condition(embeds.to(self.device), batch_size).to(
                dtype=unet_dtype
            )
            for embeds in source_pkg["uncond_image_embeds"]
        ]
        source_cond_embeds = [
            self._repeat_condition(embeds.to(self.device), batch_size).to(
                dtype=unet_dtype
            )
            for embeds in source_pkg["cond_image_embeds"]
        ]
        style_uncond_embeds = [
            self._repeat_condition(embeds, batch_size).to(dtype=unet_dtype)
            for embeds in self.uncond_image_embeds
        ]
        style_cond_embeds = [
            self._repeat_condition(embeds, batch_size).to(dtype=unet_dtype)
            for embeds in self.cond_image_embeds
        ]

        encoder_hidden_states = torch.cat(
            [
                negative_prompt_embeds,
                prompt_embeds,
                negative_prompt_embeds,
                prompt_embeds,
            ],
            dim=0,
        ).to(dtype=unet_dtype)
        add_text_embeds = torch.cat(
            [
                negative_pooled_embeds,
                pooled_embeds,
                negative_pooled_embeds,
                pooled_embeds,
            ],
            dim=0,
        ).to(dtype=unet_dtype)
        add_time_ids = torch.cat([add_time_ids] * 4, dim=0).to(
            dtype=unet_dtype
        )
        image_embeds = [
            torch.cat(
                [source_uncond, source_cond, style_uncond, style_cond], dim=0
            )
            for source_uncond, source_cond, style_uncond, style_cond in zip(
                source_uncond_embeds,
                source_cond_embeds,
                style_uncond_embeds,
                style_cond_embeds,
            )
        ]
        return encoder_hidden_states, {
            "text_embeds": add_text_embeds,
            "time_ids": add_time_ids,
            "image_embeds": image_embeds,
        }

    @staticmethod
    def _rms_clip(value, max_rms):
        if max_rms <= 0:
            return value
        rms = value.float().pow(2).mean().sqrt()
        scale = torch.clamp(value.new_tensor(max_rms) / (rms + 1e-8), max=1.0)
        return value * scale.to(dtype=value.dtype)

    def short_trajectory_distill(
        self,
        image,
        source_key,
        trajectory_steps=4,
        schedule_steps=30,
        trajectory_samples=1,
        beta=0.1,
        t_min=0.2,
        t_max=0.65,
        delta_clip=0.0,
        return_diagnostics=False,
    ):
        """Distill paired content/style DDIM trajectories into the input image."""
        if source_key not in self.source_cache:
            raise KeyError(
                f"Trajectory source_key={source_key!r} was not cached. Call "
                "cache_source_images() before optimization."
            )
        if trajectory_steps <= 0:
            raise ValueError("trajectory_steps must be positive.")
        if schedule_steps < trajectory_steps:
            raise ValueError("schedule_steps must be at least trajectory_steps.")
        if trajectory_samples <= 0:
            raise ValueError("trajectory_samples must be positive.")
        if not 0.0 <= t_min < t_max <= 1.0:
            raise ValueError("Trajectory t_min/t_max must satisfy 0 <= min < max <= 1.")

        current_latents = self._encode_latents(image)
        unet_dtype = next(self.pipe.unet.parameters()).dtype
        teacher_start = current_latents.detach().to(dtype=unet_dtype)
        batch_size = teacher_start.shape[0]
        source_pkg = self.source_cache[source_key]
        encoder_hidden_states, added_cond_kwargs = self._trajectory_condition_batch(
            source_pkg, batch_size, unet_dtype
        )

        self.trajectory_scheduler.set_timesteps(
            schedule_steps, device=self.device
        )
        schedule = self.trajectory_scheduler.timesteps
        normalized = schedule.float() / max(self.num_train_timesteps - 1, 1)
        valid = torch.nonzero(
            (normalized >= t_min) & (normalized <= t_max), as_tuple=False
        ).flatten()
        valid = valid[valid + trajectory_steps <= schedule.numel()]
        if valid.numel() == 0:
            raise ValueError(
                "No valid DDIM trajectory window exists for the requested "
                "schedule, timestep range, and trajectory length."
            )

        raw_deltas = []
        clipped_deltas = []
        content_predictions = []
        style_predictions = []
        sampled_timesteps = []

        with torch.no_grad():
            for _ in range(trajectory_samples):
                choice = torch.randint(valid.numel(), (1,), device=self.device)
                start_index = int(valid[choice].item())
                trajectory_timesteps = schedule[
                    start_index : start_index + trajectory_steps
                ]
                start_timestep = trajectory_timesteps[0]
                timestep_batch = start_timestep.expand(batch_size).long()
                noise = torch.randn_like(teacher_start)
                noisy = self.trajectory_scheduler.add_noise(
                    teacher_start, noise, timestep_batch
                )
                content_latents = noisy.clone()
                style_latents = noisy.clone()
                content_x0 = teacher_start
                style_x0 = teacher_start

                for timestep in trajectory_timesteps:
                    timestep_batch = timestep.expand(batch_size).long()
                    model_input = torch.cat(
                        [
                            content_latents,
                            content_latents,
                            style_latents,
                            style_latents,
                        ],
                        dim=0,
                    )
                    timestep_input = torch.cat([timestep_batch] * 4, dim=0)
                    model_input = self.trajectory_scheduler.scale_model_input(
                        model_input, timestep_input
                    )
                    noise_pred = self.pipe.unet(
                        model_input,
                        timestep_input,
                        encoder_hidden_states=encoder_hidden_states,
                        added_cond_kwargs=added_cond_kwargs,
                    ).sample
                    (
                        content_uncond,
                        content_cond,
                        style_uncond,
                        style_cond,
                    ) = noise_pred.chunk(4)
                    content_noise = content_uncond + self.guidance_scale * (
                        content_cond - content_uncond
                    )
                    style_noise = style_uncond + self.guidance_scale * (
                        style_cond - style_uncond
                    )

                    content_step = self.trajectory_scheduler.step(
                        content_noise,
                        timestep,
                        content_latents,
                        eta=0.0,
                        return_dict=True,
                    )
                    style_step = self.trajectory_scheduler.step(
                        style_noise,
                        timestep,
                        style_latents,
                        eta=0.0,
                        return_dict=True,
                    )
                    content_latents = content_step.prev_sample
                    style_latents = style_step.prev_sample
                    content_x0 = content_step.pred_original_sample
                    style_x0 = style_step.pred_original_sample

                raw_delta = torch.nan_to_num(style_x0 - content_x0)
                clipped_delta = self._rms_clip(raw_delta, delta_clip)
                raw_deltas.append(raw_delta)
                clipped_deltas.append(clipped_delta)
                content_predictions.append(content_x0)
                style_predictions.append(style_x0)
                sampled_timesteps.append(start_timestep.float())

        raw_delta = torch.stack(raw_deltas).mean(dim=0)
        teacher_delta = torch.stack(clipped_deltas).mean(dim=0)
        content_x0 = torch.stack(content_predictions).mean(dim=0)
        style_x0 = torch.stack(style_predictions).mean(dim=0)
        teacher_target = (
            current_latents.detach()
            + float(beta) * teacher_delta.to(dtype=current_latents.dtype)
        ).detach()
        loss = 0.5 * F.mse_loss(
            current_latents, teacher_target, reduction="sum"
        ) / max(batch_size, 1)

        if not return_diagnostics:
            return loss

        diagnostics = {
            "trajectory_timestep": torch.stack(sampled_timesteps).mean().detach(),
            "trajectory_steps": current_latents.new_tensor(float(trajectory_steps)),
            "trajectory_samples": current_latents.new_tensor(
                float(trajectory_samples)
            ),
            "trajectory_raw_delta_norm": raw_delta.float().pow(2).mean().sqrt().detach(),
            "trajectory_delta_norm": teacher_delta.float().pow(2).mean().sqrt().detach(),
            "trajectory_grad_norm": (
                float(beta) * teacher_delta.float().pow(2).mean().sqrt()
            ).detach(),
            "trajectory_teacher_gap": (
                style_x0.float() - content_x0.float()
            ).pow(2).mean().sqrt().detach(),
            "latent_norm": current_latents.detach().float().pow(2).mean().sqrt(),
        }
        teacher_package = {
            "content_latents": content_x0.detach(),
            "style_latents": style_x0.detach(),
            "target_latents": teacher_target.detach(),
        }
        return loss, diagnostics, teacher_package

    def _prepare_timesteps(self, timesteps, batch_size):
        if timesteps is None:
            return torch.randint(
                self.min_step,
                self.max_step + 1,
                (batch_size,),
                device=self.device,
                dtype=torch.long,
            )

        timesteps = torch.as_tensor(
            timesteps,
            device=self.device,
            dtype=torch.long,
        ).reshape(-1)
        if timesteps.numel() == 1:
            timesteps = timesteps.expand(batch_size)
        elif timesteps.numel() != batch_size:
            raise ValueError(
                "External CSD timesteps must contain one value or one value "
                f"per image; got {timesteps.numel()} for batch {batch_size}."
            )
        if torch.any(timesteps < 0) or torch.any(
            timesteps >= self.num_train_timesteps
        ):
            raise ValueError(
                "External CSD timesteps must lie inside the diffusion "
                f"schedule [0, {self.num_train_timesteps - 1}]."
            )
        return timesteps

    def _prepare_noise(self, latents, noise=None, noise_seed=None):
        if noise is not None and noise_seed is not None:
            raise ValueError("Pass either external CSD noise or noise_seed, not both.")
        if noise is None:
            if noise_seed is None:
                return torch.randn_like(latents)
            generator = torch.Generator(device=latents.device)
            generator.manual_seed(int(noise_seed))
            return torch.randn(
                latents.shape,
                device=latents.device,
                dtype=latents.dtype,
                generator=generator,
            )

        noise = torch.as_tensor(
            noise,
            device=latents.device,
            dtype=latents.dtype,
        )
        if noise.shape == latents.shape:
            return noise
        if (
            noise.ndim == latents.ndim
            and noise.shape[0] == 1
            and noise.shape[1:] == latents.shape[1:]
        ):
            return noise.expand_as(latents)
        raise ValueError(
            "External CSD noise shape mismatch: expected "
            f"{tuple(latents.shape)}, got {tuple(noise.shape)}."
        )

    def _forward_csd(
        self,
        image,
        source_key,
        return_diagnostics=False,
        timesteps=None,
        noise=None,
        noise_seed=None,
    ):
        if source_key not in self.source_cache:
            raise KeyError(
                f"CSD source_key={source_key!r} was not cached. Call "
                "cache_source_images() before training."
            )

        target_latents = self._encode_latents(image)
        unet_dtype = next(self.pipe.unet.parameters()).dtype
        target_latents_unet = target_latents.to(dtype=unet_dtype)
        batch_size = target_latents_unet.shape[0]
        source_pkg = self.source_cache[source_key]
        source_latents = source_pkg["latents"].to(
            device=self.device,
            dtype=unet_dtype,
        )
        source_latents = source_latents.expand(
            batch_size, *source_latents.shape[1:]
        )

        timesteps = self._prepare_timesteps(timesteps, batch_size)
        noise = self._prepare_noise(
            target_latents_unet,
            noise=noise,
            noise_seed=noise_seed,
        )
        source_noisy = self.scheduler.add_noise(source_latents, noise, timesteps)
        target_noisy = self.scheduler.add_noise(
            target_latents_unet, noise, timesteps
        )

        prompt_embeds = self._repeat_condition(self.prompt_embeds, batch_size)
        negative_prompt_embeds = self._repeat_condition(
            self.negative_prompt_embeds, batch_size
        )
        pooled_embeds = self._repeat_condition(
            self.pooled_prompt_embeds, batch_size
        )
        negative_pooled_embeds = self._repeat_condition(
            self.negative_pooled_prompt_embeds, batch_size
        )
        add_time_ids = self._repeat_condition(self.add_time_ids, batch_size)

        source_uncond_embeds = [
            self._repeat_condition(embeds.to(self.device), batch_size).to(
                dtype=unet_dtype
            )
            for embeds in source_pkg["uncond_image_embeds"]
        ]
        source_cond_embeds = [
            self._repeat_condition(embeds.to(self.device), batch_size).to(
                dtype=unet_dtype
            )
            for embeds in source_pkg["cond_image_embeds"]
        ]
        target_uncond_embeds = [
            self._repeat_condition(embeds, batch_size).to(dtype=unet_dtype)
            for embeds in self.uncond_image_embeds
        ]
        target_cond_embeds = [
            self._repeat_condition(embeds, batch_size).to(dtype=unet_dtype)
            for embeds in self.cond_image_embeds
        ]

        encoder_hidden_states = torch.cat(
            [
                negative_prompt_embeds,
                prompt_embeds,
                negative_prompt_embeds,
                prompt_embeds,
            ],
            dim=0,
        ).to(dtype=unet_dtype)
        add_text_embeds = torch.cat(
            [
                negative_pooled_embeds,
                pooled_embeds,
                negative_pooled_embeds,
                pooled_embeds,
            ],
            dim=0,
        ).to(dtype=unet_dtype)
        add_time_ids = torch.cat([add_time_ids] * 4, dim=0).to(
            dtype=unet_dtype
        )
        image_embeds = [
            torch.cat([source_uncond, source_cond, target_uncond, target_cond], dim=0)
            for source_uncond, source_cond, target_uncond, target_cond in zip(
                source_uncond_embeds,
                source_cond_embeds,
                target_uncond_embeds,
                target_cond_embeds,
            )
        ]
        model_input = torch.cat(
            [source_noisy, source_noisy, target_noisy, target_noisy], dim=0
        )
        timestep_input = torch.cat([timesteps] * 4, dim=0)
        model_input = self.scheduler.scale_model_input(
            model_input, timestep_input
        )
        added_cond_kwargs = {
            "text_embeds": add_text_embeds,
            "time_ids": add_time_ids,
            "image_embeds": image_embeds,
        }

        with torch.no_grad():
            noise_pred = self.pipe.unet(
                model_input,
                timestep_input,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
            ).sample
            (
                source_uncond,
                source_cond,
                target_uncond,
                target_cond,
            ) = noise_pred.chunk(4)
            source_uncond = self._prediction_to_epsilon(
                source_uncond, source_noisy, timesteps
            )
            source_cond = self._prediction_to_epsilon(
                source_cond, source_noisy, timesteps
            )
            target_uncond = self._prediction_to_epsilon(
                target_uncond, target_noisy, timesteps
            )
            target_cond = self._prediction_to_epsilon(
                target_cond, target_noisy, timesteps
            )

        source_direction = self.guidance_scale * (source_cond - source_uncond)
        target_direction = self.guidance_scale * (target_cond - target_uncond)
        weight = (1.0 - self.alphas_cumprod[timesteps]).view(
            batch_size, 1, 1, 1
        )
        grad = weight.to(target_direction.dtype) * (
            target_direction - source_direction
        )
        if self.grad_clip > 0:
            grad = grad.clamp(-self.grad_clip, self.grad_clip)
        grad = torch.nan_to_num(grad)

        loss = (grad.detach().to(target_latents.dtype) * target_latents).sum() / max(
            batch_size, 1
        )
        if not return_diagnostics:
            return loss

        source_gap = (source_cond.float() - source_uncond.float()).pow(2).mean().sqrt()
        target_gap = (target_cond.float() - target_uncond.float()).pow(2).mean().sqrt()
        diagnostics = {
            "timestep": timesteps.detach().float().mean(),
            "condition_gap": target_gap.detach(),
            "source_condition_gap": source_gap.detach(),
            "target_condition_gap": target_gap.detach(),
            "csd_direction_norm": (
                target_direction.float() - source_direction.float()
            ).pow(2).mean().sqrt().detach(),
            "sds_grad_norm": grad.detach().float().pow(2).mean().sqrt(),
            "csd_grad_norm": grad.detach().float().pow(2).mean().sqrt(),
            "latent_norm": target_latents.detach().float().pow(2).mean().sqrt(),
            "guided_noise_norm": target_direction.detach().float().pow(2).mean().sqrt(),
        }
        return loss, diagnostics

    def forward(
        self,
        image,
        source_key=None,
        return_diagnostics=False,
        timesteps=None,
        noise=None,
        noise_seed=None,
    ):
        if self.gradient_mode == "csd":
            return self._forward_csd(
                image,
                source_key,
                return_diagnostics,
                timesteps=timesteps,
                noise=noise,
                noise_seed=noise_seed,
            )

        if timesteps is not None or noise is not None or noise_seed is not None:
            raise ValueError(
                "External timestep/noise scheduling is currently supported by "
                "CSD guidance only."
            )

        latents = self._encode_latents(image)
        unet_dtype = next(self.pipe.unet.parameters()).dtype
        latents_unet = latents.to(dtype=unet_dtype)
        batch_size = latents_unet.shape[0]

        timesteps = torch.randint(
            self.min_step,
            self.max_step + 1,
            (batch_size,),
            device=self.device,
            dtype=torch.long,
        )
        noise = torch.randn_like(latents_unet)
        noisy_latents = self.scheduler.add_noise(latents_unet, noise, timesteps)
        do_cfg = self.guidance_scale > 1.0

        prompt_embeds = self._repeat_condition(self.prompt_embeds, batch_size)
        pooled_embeds = self._repeat_condition(
            self.pooled_prompt_embeds, batch_size
        )
        add_time_ids = self._repeat_condition(self.add_time_ids, batch_size)
        cond_image_embeds = [
            self._repeat_condition(embeds, batch_size).to(dtype=unet_dtype)
            for embeds in self.cond_image_embeds
        ]

        if do_cfg:
            negative_prompt_embeds = self._repeat_condition(
                self.negative_prompt_embeds, batch_size
            )
            negative_pooled_embeds = self._repeat_condition(
                self.negative_pooled_prompt_embeds, batch_size
            )
            uncond_image_embeds = [
                self._repeat_condition(embeds, batch_size).to(dtype=unet_dtype)
                for embeds in self.uncond_image_embeds
            ]
            encoder_hidden_states = torch.cat(
                [negative_prompt_embeds, prompt_embeds], dim=0
            )
            add_text_embeds = torch.cat(
                [negative_pooled_embeds, pooled_embeds], dim=0
            )
            add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)
            image_embeds = [
                torch.cat([uncond, cond], dim=0)
                for uncond, cond in zip(uncond_image_embeds, cond_image_embeds)
            ]
            model_input = torch.cat([noisy_latents, noisy_latents], dim=0)
            timestep_input = torch.cat([timesteps, timesteps], dim=0)
        else:
            encoder_hidden_states = prompt_embeds
            add_text_embeds = pooled_embeds
            image_embeds = cond_image_embeds
            model_input = noisy_latents
            timestep_input = timesteps

        model_input = self.scheduler.scale_model_input(model_input, timestep_input)
        added_cond_kwargs = {
            "text_embeds": add_text_embeds.to(dtype=unet_dtype),
            "time_ids": add_time_ids.to(dtype=unet_dtype),
            "image_embeds": image_embeds,
        }
        condition_gap = latents_unet.new_zeros(())
        with torch.no_grad():
            noise_pred = self.pipe.unet(
                model_input,
                timestep_input,
                encoder_hidden_states=encoder_hidden_states.to(dtype=unet_dtype),
                added_cond_kwargs=added_cond_kwargs,
            ).sample
            if do_cfg:
                noise_uncond, noise_style = noise_pred.chunk(2)
                condition_gap = (
                    (noise_style.float() - noise_uncond.float())
                    .pow(2)
                    .mean()
                    .sqrt()
                )
                noise_pred = noise_uncond + self.guidance_scale * (
                    noise_style - noise_uncond
                )
            noise_pred = self._prediction_to_epsilon(
                noise_pred,
                noisy_latents,
                timesteps,
            )

        weight = (1.0 - self.alphas_cumprod[timesteps]).view(
            batch_size, 1, 1, 1
        )
        grad = weight.to(noise_pred.dtype) * (noise_pred - noise)
        if self.grad_clip > 0:
            grad = grad.clamp(-self.grad_clip, self.grad_clip)
        grad = torch.nan_to_num(grad)

        # The detached score supplies dL/dz; only the VAE input remains trainable.
        loss = (grad.detach().to(latents.dtype) * latents).sum() / max(
            batch_size, 1
        )
        if not return_diagnostics:
            return loss

        diagnostics = {
            "timestep": timesteps.detach().float().mean(),
            "condition_gap": condition_gap.detach().float(),
            "sds_grad_norm": grad.detach().float().pow(2).mean().sqrt(),
            "latent_norm": latents.detach().float().pow(2).mean().sqrt(),
            "guided_noise_norm": noise_pred.detach().float().pow(2).mean().sqrt(),
        }
        return loss, diagnostics
