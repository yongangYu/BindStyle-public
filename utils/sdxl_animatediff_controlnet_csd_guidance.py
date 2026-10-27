import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from utils.mesh_renderer_nvdiffrast import NVDiffMeshRenderer, sequence_inverse_depth_range
from utils.style_numerics import (
    check_backward_finite,
    require_finite,
    video_csd_surrogate,
)


class SDXLAnimateDiffControlNetCSDGuidance(nn.Module):
    """Frozen video diffusion teacher for paired, window-level CSD.

    The source branches pair the original RGB sequence with a pooled content
    image condition. The target branches pair Gaussian renders with one shared
    IP-Adapter style condition. Both sides use the same GT-mesh depth controls,
    motion model, diffusion timestep, and noise tensor.
    """

    enabled = True
    expects_video = True
    manages_device_placement = True

    def __init__(self, cfg, device):
        super().__init__()
        try:
            from diffusers import (
                AnimateDiffSDXLPipeline,
                ControlNetModel,
                DDPMScheduler,
                MotionAdapter,
                StableDiffusionXLPipeline,
            )
            from transformers import CLIPVisionModelWithProjection
        except (ImportError, RuntimeError) as exc:
            raise ImportError(
                "SDXL video CSD requires a Diffusers build containing "
                "AnimateDiffSDXLPipeline, MotionAdapter, and ControlNetModel."
            ) from exc

        self.device = torch.device(device)
        self.temporal_mode = str(cfg.get("temporal_mode", "motion"))
        if self.temporal_mode not in {"motion", "framewise"}:
            raise ValueError("temporal_mode must be motion or framewise")
        self.model_name = str(
            cfg.get("model_name", "stabilityai/stable-diffusion-xl-base-1.0")
        )
        self.motion_adapter_name = str(
            cfg.get(
                "motion_adapter_name",
                "guoyww/animatediff-motion-adapter-sdxl-beta",
            )
        )
        self.controlnet_model_name = str(
            cfg.get(
                "controlnet_model_name",
                "diffusers/controlnet-depth-sdxl-1.0",
            )
        )
        self.ip_adapter_name = str(cfg.get("ip_adapter_name", "h94/IP-Adapter"))
        self.ip_adapter_subfolder = str(
            cfg.get("ip_adapter_subfolder", "sdxl_models")
        )
        self.ip_adapter_weight_name = str(
            cfg.get("ip_adapter_weight_name", "ip-adapter_sdxl.bin")
        )
        self.image_encoder_folder = str(
            cfg.get("image_encoder_folder", "image_encoder")
        )
        self.style_path = str(cfg.get("style_path", ""))
        if not self.style_path:
            raise ValueError("A style_path is required for SDXL video CSD.")
        if not os.path.isfile(self.style_path):
            raise FileNotFoundError(f"Style image was not found: {self.style_path}")

        self.cache_dir = cfg.get("cache_dir", None) or None
        self.local_files_only = bool(cfg.get("local_files_only", False))
        self.revision = cfg.get("revision", None) or None
        self.variant = cfg.get("variant", None) or None
        self.motion_adapter_variant = (
            cfg.get("motion_adapter_variant", None) or None
        )
        self.controlnet_variant = cfg.get("controlnet_variant", None) or None
        self.token = cfg.get("token", None) or None
        self.prompt = str(cfg.get("prompt", ""))
        self.negative_prompt = str(cfg.get("negative_prompt", ""))
        self.gradient_mode = "csd"
        self.guidance_scale = float(cfg.get("guidance_scale", 5.0))
        self.ip_adapter_scale = float(cfg.get("ip_adapter_scale", 0.8))
        self.controlnet_conditioning_scale = float(
            cfg.get("controlnet_conditioning_scale", 0.7)
        )
        self.control_depth_mode = str(cfg.get("control_depth_mode", "ndc"))
        self.window_size = max(2, int(cfg.get("video_window_size", 4)))
        self.image_size = int(cfg.get("image_size", 512))
        if self.image_size <= 0 or self.image_size % 8 != 0:
            raise ValueError("image_size must be a positive multiple of 8.")
        self.t_min = float(cfg.get("t_min", 0.2))
        self.t_max = float(cfg.get("t_max", 0.65))
        self.grad_clip = float(cfg.get("sds_grad_clip", 1.0))
        self.vae_latent_mode = str(cfg.get("vae_latent_mode", "mode")).lower()
        self.offload_condition_encoders = bool(
            cfg.get("offload_condition_encoders", True)
        )

        self._validate_local_paths()
        torch_dtype = (
            torch.float16 if bool(cfg.get("half_precision", True)) else torch.float32
        )
        common_kwargs = {
            "torch_dtype": torch_dtype,
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only,
        }
        if self.token:
            common_kwargs["token"] = self.token

        motion_kwargs = dict(common_kwargs)
        if self.motion_adapter_variant:
            motion_kwargs["variant"] = self.motion_adapter_variant
        motion_adapter = None
        if self.temporal_mode == "motion":
            motion_adapter = MotionAdapter.from_pretrained(
                self.motion_adapter_name, **motion_kwargs
            )

        model_kwargs = dict(common_kwargs)
        if self.revision:
            model_kwargs["revision"] = self.revision
        if self.variant:
            model_kwargs["variant"] = self.variant
        if self.temporal_mode == "motion":
            self.pipe = AnimateDiffSDXLPipeline.from_pretrained(
                self.model_name, motion_adapter=motion_adapter, **model_kwargs,
            )
        else:
            self.pipe = StableDiffusionXLPipeline.from_pretrained(self.model_name, **model_kwargs)

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
        self.pipe.enable_vae_slicing()
        self.pipe.enable_vae_tiling()
        if hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing("max")

        control_kwargs = dict(common_kwargs)
        if self.controlnet_variant:
            control_kwargs["variant"] = self.controlnet_variant
        self.controlnet = ControlNetModel.from_pretrained(
            self.controlnet_model_name, **control_kwargs
        )
        unet_channels = int(self.pipe.unet.config.in_channels)
        controlnet_channels = int(self.controlnet.config.in_channels)
        if unet_channels != controlnet_channels:
            raise ValueError(
                "The SDXL Motion UNet and ControlNet input channels do not "
                f"match ({unet_channels} != {controlnet_channels})."
            )
        motion_max_frames = getattr(
            self.pipe.unet.config, "motion_max_seq_length", None
        )
        if motion_max_frames is not None and self.window_size > int(
            motion_max_frames
        ):
            raise ValueError(
                "video_window_size exceeds the motion adapter limit "
                f"({self.window_size} > {motion_max_frames})."
            )

        for module_name in [
            "vae",
            "unet",
            "motion_adapter",
            "text_encoder",
            "text_encoder_2",
            "image_encoder",
        ]:
            module = getattr(self.pipe, module_name, None)
            if module is None:
                continue
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.controlnet.eval()
        for parameter in self.controlnet.parameters():
            parameter.requires_grad_(False)

        # Prompt and reference-image conditions are computed once. Keep the
        # large denoisers on CPU during this phase so a 16 GB GPU never has to
        # hold SDXL, both condition encoders, and ControlNet simultaneously.
        for module_name in ["text_encoder", "text_encoder_2", "image_encoder"]:
            module = getattr(self.pipe, module_name, None)
            if module is not None:
                module.to(self.device)

        self.scheduler = DDPMScheduler.from_config(self.pipe.scheduler.config)
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        self.num_train_timesteps = int(self.scheduler.config.num_train_timesteps)
        self.min_step = max(0, int(self.num_train_timesteps * self.t_min))
        self.max_step = min(
            self.num_train_timesteps - 1,
            int(self.num_train_timesteps * self.t_max),
        )
        if self.max_step <= self.min_step:
            raise ValueError("t_max must be greater than t_min.")
        self.register_buffer(
            "alphas_cumprod",
            self.scheduler.alphas_cumprod.detach().float(),
        )

        self.source_cache = {}
        self.control_cache = {}
        self._encode_style_condition()
        if self.offload_condition_encoders:
            for module_name in ["text_encoder", "text_encoder_2", "image_encoder"]:
                module = getattr(self.pipe, module_name, None)
                if module is not None:
                    module.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self._configure_vae_precision(
            torch_dtype, bool(cfg.get("vae_half_precision", False))
        )
        self._backbone_active = False

    def _configure_vae_precision(self, model_dtype, half_precision):
        requested_dtype = model_dtype if half_precision else torch.float32
        force_upcast = bool(getattr(self.pipe.vae.config, "force_upcast", True))
        self.vae_dtype = torch.float32 if force_upcast else requested_dtype
        if self.vae_dtype != requested_dtype:
            print(
                "[MaGS] VAE force_upcast=true: using float32 for differentiable "
                "encoding despite vae_half_precision=true."
            )
        self.pipe.vae.to(device=self.device, dtype=self.vae_dtype)
        print(f"[MaGS] Video CSD VAE encoding dtype: {self.vae_dtype}.")

    def _activate_diffusion_backbone(self):
        if self._backbone_active:
            return
        self.pipe.unet.to(self.device)
        self.controlnet.to(self.device)
        self._backbone_active = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not getattr(self, "_activation_logged", False):
            print("[MaGS] Activated UNetMotion + ControlNet after condition caching.")
            self._activation_logged = True

    def park_diffusion_backbone(self):
        """Free denoiser weight memory while the independent VGG branch runs."""
        if not self._backbone_active:
            return
        self.pipe.unet.to("cpu")
        self.controlnet.to("cpu")
        self._backbone_active = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _local_image_encoder_dir(self):
        if not self.image_encoder_folder or not os.path.isdir(self.ip_adapter_name):
            return None
        adapter_dir = self.ip_adapter_name
        if self.ip_adapter_subfolder:
            adapter_dir = os.path.join(adapter_dir, self.ip_adapter_subfolder)
        return os.path.join(adapter_dir, self.image_encoder_folder)

    @staticmethod
    def _require_local_component(path, label, marker="config.json"):
        if not os.path.isabs(path):
            return
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Local {label} directory does not exist: {path}")
        if marker and not os.path.isfile(os.path.join(path, marker)):
            raise FileNotFoundError(
                f"Local {label} directory is incomplete; {marker} was not found: {path}"
            )

    def _validate_local_paths(self):
        self._require_local_component(self.model_name, "SDXL", "model_index.json")
        if self.temporal_mode == "motion":
            self._require_local_component(self.motion_adapter_name, "AnimateDiff motion adapter")
        self._require_local_component(self.controlnet_model_name, "SDXL ControlNet")
        if not os.path.isabs(self.ip_adapter_name):
            return
        self._require_local_component(self.ip_adapter_name, "IP-Adapter", None)
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
            self._require_local_component(image_encoder_dir, "IP-Adapter image encoder")

    @torch.no_grad()
    def _encode_style_condition(self):
        encode_kwargs = {
            "prompt": self.prompt,
            "prompt_2": self.prompt,
            "device": self.device,
            "do_classifier_free_guidance": True,
            "negative_prompt": self.negative_prompt,
            "negative_prompt_2": self.negative_prompt,
        }
        try:
            encoded = self.pipe.encode_prompt(
                num_videos_per_prompt=1, **encode_kwargs
            )
        except TypeError:
            encoded = self.pipe.encode_prompt(
                num_images_per_prompt=1, **encode_kwargs
            )
        (
            self.prompt_embeds,
            self.negative_prompt_embeds,
            self.pooled_prompt_embeds,
            self.negative_pooled_prompt_embeds,
        ) = [value.detach() for value in encoded[:4]]

        style_image = Image.open(self.style_path).convert("RGB")
        image_embeds = self.pipe.prepare_ip_adapter_image_embeds(
            ip_adapter_image=style_image,
            ip_adapter_image_embeds=None,
            device=self.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )
        self.uncond_image_embeds = []
        self.cond_image_embeds = []
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

    def _encode_frame_batch(self, images):
        require_finite(images, "VAE input image")
        images = F.interpolate(
            images.float(),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)
        vae_dtype = next(self.pipe.vae.parameters()).dtype
        image_model = (images * 2.0 - 1.0).to(self.device, dtype=vae_dtype)
        check_backward_finite(image_model, "VAE backward image gradient")
        with torch.autocast(device_type=self.device.type, enabled=False):
            latent_dist = self.pipe.vae.encode(image_model).latent_dist
            if self.vae_latent_mode == "sample":
                latents = latent_dist.sample()
            else:
                try:
                    latents = latent_dist.mode()
                except AttributeError:
                    latents = latent_dist.mean
            latents = latents.float() * self.pipe.vae.config.scaling_factor
        require_finite(latents, "VAE encoded latents")
        return check_backward_finite(latents, "VAE latent gradient")

    def _encode_video_latents(self, video):
        if video.dim() != 5 or video.shape[2] != 3:
            raise ValueError("Video must have shape [B, F, 3, H, W].")
        batch_size, frame_count = video.shape[:2]
        frames = video.reshape(batch_size * frame_count, *video.shape[2:])
        latents = self._encode_frame_batch(frames)
        return latents.reshape(
            batch_size, frame_count, *latents.shape[1:]
        ).permute(0, 2, 1, 3, 4).contiguous()

    @staticmethod
    def _image_to_batch(image):
        if not torch.is_tensor(image):
            raise TypeError("Source frames must be torch tensors.")
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.dim() != 4 or image.shape[1] != 3:
            raise ValueError("Source frames must have shape [B, 3, H, W].")
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
        if not isinstance(source_images, dict):
            source_images = dict(source_images)
        image_encoder = getattr(self.pipe, "image_encoder", None)
        if image_encoder is None:
            raise RuntimeError("Video CSD requires the IP-Adapter image encoder.")
        image_encoder.to(self.device)
        for key, source_image in source_images.items():
            source_batch = self._image_to_batch(source_image).to(self.device)
            source_embeds = self.pipe.prepare_ip_adapter_image_embeds(
                ip_adapter_image=self._batch_to_pil(source_batch),
                ip_adapter_image_embeds=None,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
            )
            source_uncond = []
            source_cond = []
            for embeds in source_embeds:
                uncond, cond = embeds.chunk(2, dim=0)
                source_uncond.append(uncond.detach().cpu())
                source_cond.append(cond.detach().cpu())
            self.source_cache[key] = {
                "latents": self._encode_frame_batch(source_batch).detach().cpu(),
                "uncond_image_embeds": source_uncond,
                "cond_image_embeds": source_cond,
            }
        if self.offload_condition_encoders:
            image_encoder.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(
            f"[MaGS] Cached {len(self.source_cache)} RGB/content conditions "
            "for video CSD."
        )

    @torch.no_grad()
    def cache_control_frames(self, cameras):
        for cam in cameras:
            if cam.mesh_vertices is None or cam.mesh_faces is None:
                raise ValueError("Video ControlNet guidance requires a GT mesh for every frame.")
        depth_range = (
            sequence_inverse_depth_range(cameras)
            if self.control_depth_mode == "relative_inverse" else None
        )
        renderer = NVDiffMeshRenderer(
            resolution=self.image_size,
            mode="depth",
            background=(0.0, 0.0, 0.0),
            antialias=True,
            depth_mode=self.control_depth_mode,
            depth_range=depth_range,
        ).to(self.device)
        for cam in cameras:
            if cam.mesh_vertices is None or cam.mesh_faces is None:
                raise ValueError(
                    "Video ControlNet guidance requires a GT mesh for every frame."
                )
            package = renderer(
                cam.mesh_vertices.to(self.device),
                cam.mesh_faces.to(self.device),
                cam,
            )
            self.control_cache[cam.uid] = package["image"].detach().cpu()
        print(
            f"[MaGS] Cached {len(self.control_cache)} GT-mesh depth controls "
            f"for video CSD (mode={self.control_depth_mode}, range={depth_range})."
        )
        self._activate_diffusion_backbone()

    @staticmethod
    def _repeat(condition, batch_size):
        return condition.expand(batch_size, *condition.shape[1:])

    def _source_video_latents(self, source_keys, dtype):
        latents = []
        for key in source_keys:
            if key not in self.source_cache:
                raise KeyError(f"Video CSD source frame was not cached: {key!r}")
            latents.append(self.source_cache[key]["latents"].to(self.device, dtype=dtype))
        return torch.stack([value[0] for value in latents], dim=1).unsqueeze(0)

    def _source_video_image_embeds(self, source_keys, dtype):
        """Pool frame-wise CLIP conditions into one content condition per video."""
        uncond_embeds = []
        cond_embeds = []
        adapter_count = len(self.uncond_image_embeds)
        for adapter_index in range(adapter_count):
            frame_uncond = []
            frame_cond = []
            for key in source_keys:
                package = self.source_cache[key]
                frame_uncond.append(
                    package["uncond_image_embeds"][adapter_index].to(
                        self.device, dtype=dtype
                    )
                )
                frame_cond.append(
                    package["cond_image_embeds"][adapter_index].to(
                        self.device, dtype=dtype
                    )
                )
            uncond_embeds.append(torch.stack(frame_uncond, dim=0).mean(dim=0))
            cond_embeds.append(torch.stack(frame_cond, dim=0).mean(dim=0))
        return uncond_embeds, cond_embeds

    def _control_video(self, source_keys, dtype):
        controls = []
        for key in source_keys:
            if key not in self.control_cache:
                raise KeyError(f"Video CSD depth control was not cached: {key!r}")
            controls.append(self.control_cache[key][0].to(self.device, dtype=dtype))
        return torch.stack(controls, dim=0).unsqueeze(0)

    def _prepare_timesteps(self, timesteps, batch_size):
        if timesteps is None:
            return torch.randint(
                self.min_step,
                self.max_step + 1,
                (batch_size,),
                device=self.device,
                dtype=torch.long,
            )
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(timesteps, device=self.device, dtype=torch.long)
        timesteps = timesteps.to(self.device, dtype=torch.long).reshape(-1)
        if timesteps.numel() == 1:
            timesteps = timesteps.expand(batch_size)
        if timesteps.numel() != batch_size:
            raise ValueError("Video CSD requires one timestep per video batch.")
        return timesteps

    def _prepare_noise(self, latents, noise=None, noise_seed=None):
        if noise is not None and noise_seed is not None:
            raise ValueError("Pass either noise or noise_seed, not both.")
        if noise is not None:
            noise = noise.to(self.device, dtype=latents.dtype)
            if noise.shape == latents.shape:
                return noise
            if noise.shape[0] == 1 and noise.shape[1:] == latents.shape[1:]:
                return noise.expand_as(latents)
            raise ValueError(
                "External video CSD noise shape mismatch: expected "
                f"{tuple(latents.shape)}, got {tuple(noise.shape)}."
            )
        if noise_seed is None:
            return torch.randn_like(latents)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(noise_seed))
        return torch.randn(
            latents.shape,
            generator=generator,
            device=self.device,
            dtype=latents.dtype,
        )

    def _prediction_to_epsilon(self, prediction, noisy_latents, timesteps):
        prediction = prediction.float()
        noisy_latents = noisy_latents.float()
        prediction_type = getattr(self.scheduler.config, "prediction_type", "epsilon")
        if prediction_type == "epsilon":
            return prediction
        if prediction_type == "v_prediction":
            alpha = self.alphas_cumprod[timesteps].view(-1, 1, 1, 1, 1)
            beta = 1.0 - alpha
            return alpha.sqrt() * prediction + beta.sqrt() * noisy_latents
        return prediction

    def _branch_conditions(self, batch_size, conditional, image_embeds, dtype):
        if conditional:
            prompt = self._repeat(self.prompt_embeds, batch_size)
            pooled = self._repeat(self.pooled_prompt_embeds, batch_size)
        else:
            prompt = self._repeat(self.negative_prompt_embeds, batch_size)
            pooled = self._repeat(self.negative_pooled_prompt_embeds, batch_size)
        image_embeds = [
            self._repeat(value, batch_size).to(self.device, dtype=dtype)
            for value in image_embeds
        ]
        time_ids = self._repeat(self.add_time_ids, batch_size)
        return (
            prompt.to(self.device, dtype=dtype),
            pooled.to(self.device, dtype=dtype),
            time_ids.to(self.device, dtype=dtype),
            image_embeds,
        )

    @torch.no_grad()
    def _predict_video_epsilon(
        self,
        noisy,
        timesteps,
        control_video,
        conditional,
        image_embeds,
    ):
        unet_dtype = next(self.pipe.unet.parameters()).dtype
        control_dtype = next(self.controlnet.parameters()).dtype
        noisy = noisy.to(dtype=unet_dtype)
        batch_size, _, frame_count, height, width = noisy.shape
        prompt, pooled, time_ids, image_embeds = self._branch_conditions(
            batch_size, conditional, image_embeds, unet_dtype
        )
        model_input = self.scheduler.scale_model_input(noisy, timesteps)
        require_finite(model_input, "UNetMotion/ControlNet noisy input")

        # UNetMotionModel flattens spatial processing to B*F. Diffusers'
        # AnimateDiff pipelines therefore repeat text tokens once per frame,
        # while pooled SDXL and IP-Adapter conditions remain video-level B.
        frame_prompt = prompt.repeat_interleave(frame_count, dim=0)

        control_input = model_input.permute(0, 2, 1, 3, 4).reshape(
            batch_size * frame_count, -1, height, width
        ).to(dtype=control_dtype)
        control_timesteps = timesteps[:, None].expand(
            batch_size, frame_count
        ).reshape(-1)
        control_prompt = frame_prompt.to(control_dtype)
        control_pooled = pooled[:, None].expand(
            batch_size, frame_count, pooled.shape[-1]
        ).reshape(batch_size * frame_count, pooled.shape[-1]).to(control_dtype)
        control_time_ids = time_ids[:, None].expand(
            batch_size, frame_count, time_ids.shape[-1]
        ).reshape(batch_size * frame_count, time_ids.shape[-1]).to(control_dtype)
        control_cond = control_video.expand(
            batch_size, *control_video.shape[1:]
        ).reshape(
            batch_size * frame_count,
            *control_video.shape[2:],
        ).to(dtype=control_dtype)

        down_residuals, mid_residual = self.controlnet(
            control_input,
            control_timesteps,
            encoder_hidden_states=control_prompt,
            controlnet_cond=control_cond,
            conditioning_scale=self.controlnet_conditioning_scale,
            added_cond_kwargs={
                "text_embeds": control_pooled,
                "time_ids": control_time_ids,
            },
            guess_mode=False,
            return_dict=False,
        )
        down_residuals = tuple(
            residual.to(dtype=unet_dtype) for residual in down_residuals
        )
        mid_residual = mid_residual.to(dtype=unet_dtype)
        for index, residual in enumerate(down_residuals):
            require_finite(residual, f"ControlNet down residual {index}")
        require_finite(mid_residual, "ControlNet mid residual")
        if self.temporal_mode == "framewise":
            # The original 2D SDXL UNet has no temporal modules. Conditions and
            # residuals follow the same B-major, F-minor frame ordering.
            prediction = self.pipe.unet(
                control_input.to(unet_dtype), control_timesteps,
                encoder_hidden_states=frame_prompt,
                added_cond_kwargs={
                    "text_embeds": control_pooled.to(unet_dtype),
                    "time_ids": control_time_ids.to(unet_dtype),
                    "image_embeds": [value.repeat_interleave(frame_count, dim=0) for value in image_embeds],
                },
                down_block_additional_residuals=down_residuals,
                mid_block_additional_residual=mid_residual,
                return_dict=False,
            )[0]
            prediction = prediction.reshape(batch_size, frame_count, *prediction.shape[1:]).permute(0, 2, 1, 3, 4)
            return self._prediction_to_epsilon(prediction, noisy, timesteps)
        prediction = self.pipe.unet(
            model_input,
            timesteps,
            encoder_hidden_states=frame_prompt,
            added_cond_kwargs={
                "text_embeds": pooled,
                "time_ids": time_ids,
                "image_embeds": image_embeds,
            },
            down_block_additional_residuals=down_residuals,
            mid_block_additional_residual=mid_residual,
            return_dict=False,
        )[0]
        return self._prediction_to_epsilon(prediction, noisy, timesteps)

    def forward(
        self,
        image,
        source_key=None,
        source_keys=None,
        return_diagnostics=False,
        timesteps=None,
        noise=None,
        noise_seed=None,
    ):
        self._activate_diffusion_backbone()
        if source_keys is None:
            source_keys = source_key
        if not isinstance(source_keys, (list, tuple)):
            raise TypeError("Video CSD requires source_keys as an ordered list.")
        if image.dim() != 5:
            raise ValueError("Video CSD input must have shape [B, F, 3, H, W].")
        if image.shape[0] != 1:
            raise ValueError("The first implementation supports video batch size 1.")
        if image.shape[1] != len(source_keys):
            raise ValueError("Video frame count and source_keys length do not match.")
        if image.shape[1] != self.window_size:
            raise ValueError(
                "Video CSD input frame count must equal video_window_size "
                f"({image.shape[1]} != {self.window_size})."
            )

        target_latents = self._encode_video_latents(image)
        unet_dtype = next(self.pipe.unet.parameters()).dtype
        target_unet = target_latents.to(dtype=unet_dtype)
        source_latents = self._source_video_latents(source_keys, unet_dtype)
        source_uncond_embeds, source_cond_embeds = (
            self._source_video_image_embeds(source_keys, unet_dtype)
        )
        control_video = self._control_video(source_keys, unet_dtype)
        timesteps = self._prepare_timesteps(timesteps, target_unet.shape[0])
        noise = self._prepare_noise(target_unet, noise=noise, noise_seed=noise_seed)
        source_noisy = self.scheduler.add_noise(source_latents, noise, timesteps)
        target_noisy = self.scheduler.add_noise(target_unet, noise, timesteps)

        source_uncond = self._predict_video_epsilon(
            source_noisy,
            timesteps,
            control_video,
            conditional=False,
            image_embeds=source_uncond_embeds,
        )
        source_cond = self._predict_video_epsilon(
            source_noisy,
            timesteps,
            control_video,
            conditional=True,
            image_embeds=source_cond_embeds,
        )
        target_uncond = self._predict_video_epsilon(
            target_noisy,
            timesteps,
            control_video,
            conditional=False,
            image_embeds=self.uncond_image_embeds,
        )
        target_cond = self._predict_video_epsilon(
            target_noisy,
            timesteps,
            control_video,
            conditional=True,
            image_embeds=self.cond_image_embeds,
        )
        weight = (1.0 - self.alphas_cumprod[timesteps]).view(-1, 1, 1, 1, 1)
        loss, grad, source_direction, target_direction = video_csd_surrogate(
            target_latents,
            source_uncond,
            source_cond,
            target_uncond,
            target_cond,
            weight,
            self.guidance_scale,
            self.grad_clip,
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
            "video_frames": target_latents.new_tensor(float(image.shape[1])),
            "controlnet_scale": target_latents.new_tensor(
                self.controlnet_conditioning_scale
            ),
            "ip_adapter_scale": target_latents.new_tensor(self.ip_adapter_scale),
        }
        return loss, diagnostics
