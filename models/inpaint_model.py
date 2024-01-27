from diffusers import  UNet2DConditionModel, AutoencoderKL, DDPMScheduler, DDIMScheduler, RePaintScheduler, StableDiffusionInpaintPipeline,EulerAncestralDiscreteScheduler, ControlNetModel, T2IAdapter
from diffusers.image_processor import VaeImageProcessor
from transformers import CLIPImageProcessor, CLIPTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPTextModel
import torch
from diffusers.utils.torch_utils import randn_tensor
from .modules import get_cartesian, CusTransformer, CLIPVisionEmbeddings, CrossAttnStoreProcessor, FeedForward
import numpy as np
from einops import rearrange
import random
import pytorch_lightning as pl
import torchvision.transforms as T
from diffusers.utils.import_utils import is_xformers_available



def prepare_mask_and_masked_image(image, mask):
    """
    Prepares a pair (image, mask) to be consumed by the Stable Diffusion pipeline. This means that those inputs will be
    converted to ``torch.Tensor`` with shapes ``batch x channels x height x width`` where ``channels`` is ``3`` for the
    ``image`` and ``1`` for the ``mask``.

    The ``image`` will be converted to ``torch.float32`` and normalized to be in ``[-1, 1]``. The ``mask`` will be
    binarized (``mask > 0.5``) and cast to ``torch.float32`` too.

    Args:
        image (Union[np.array, PIL.Image, torch.Tensor]): The image to inpaint.
            It can be a ``PIL.Image``, or a ``height x width x 3`` ``np.array`` or a ``channels x height x width``
            ``torch.Tensor`` or a ``batch x channels x height x width`` ``torch.Tensor``.
        mask (_type_): The mask to apply to the image, i.e. regions to inpaint.
            It can be a ``PIL.Image``, or a ``height x width`` ``np.array`` or a ``1 x height x width``
            ``torch.Tensor`` or a ``batch x 1 x height x width`` ``torch.Tensor``.


    Raises:
        ValueError: ``torch.Tensor`` images should be in the ``[-1, 1]`` range. ValueError: ``torch.Tensor`` mask
        should be in the ``[0, 1]`` range. ValueError: ``mask`` and ``image`` should have the same spatial dimensions.
        TypeError: ``mask`` is a ``torch.Tensor`` but ``image`` is not
            (ot the other way around).

    Returns:
        tuple[torch.Tensor]: The pair (mask, masked_image) as ``torch.Tensor`` with 4
            dimensions: ``batch x channels x height x width``.
    """
    if isinstance(image, torch.Tensor):
        if not isinstance(mask, torch.Tensor):
            raise TypeError(f"`image` is a torch.Tensor but `mask` (type: {type(mask)} is not")

        # Batch single image
        if image.ndim == 3:
            assert image.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
            image = image.unsqueeze(0)

        # Batch and add channel dim for single mask
        if mask.ndim == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)

        # Batch single mask or add channel dim
        if mask.ndim == 3:
            # Single batched mask, no channel dim or single mask not batched but channel dim
            if mask.shape[0] == 1:
                mask = mask.unsqueeze(0)

            # Batched masks no channel dim
            else:
                mask = mask.unsqueeze(1)

        assert image.ndim == 4 and mask.ndim == 4, "Image and Mask must have 4 dimensions"
        assert image.shape[-2:] == mask.shape[-2:], "Image and Mask must have the same spatial dimensions"
        assert image.shape[0] == mask.shape[0], "Image and Mask must have the same batch size"

        # Check image is in [-1, 1]
        if image.min() < -1 or image.max() > 1:
            raise ValueError(f"Image should be in [-1, 1] range, but image is in [{image.min()}, {image.max()}]")

        # Check mask is in [0, 1]
        if mask.min() < 0 or mask.max() > 1:
            raise ValueError("Mask should be in [0, 1] range")

        # Binarize mask
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        # Image as float32
        image = image.to(dtype=torch.float32)
    elif isinstance(mask, torch.Tensor):
        raise TypeError(f"`mask` is a torch.Tensor but `image` (type: {type(image)} is not")
    else:
        # preprocess image
        if isinstance(image, (PIL.Image.Image, np.ndarray)):
            image = [image]

        if isinstance(image, list) and isinstance(image[0], PIL.Image.Image):
            image = [np.array(i.convert("RGB"))[None, :] for i in image]
            image = np.concatenate(image, axis=0)
        elif isinstance(image, list) and isinstance(image[0], np.ndarray):
            image = np.concatenate([i[None, :] for i in image], axis=0)

        image = image.transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

        # preprocess mask
        if isinstance(mask, (PIL.Image.Image, np.ndarray)):
            mask = [mask]

        if isinstance(mask, list) and isinstance(mask[0], PIL.Image.Image):
            mask = np.concatenate([np.array(m.convert("L"))[None, None, :] for m in mask], axis=0)
            mask = mask.astype(np.float32) / 255.0
        elif isinstance(mask, list) and isinstance(mask[0], np.ndarray):
            mask = np.concatenate([m[None, None, :] for m in mask], axis=0)

        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    return mask, masked_image

def add_noise(latents, scheduler, timesteps=None, noise=None, use_offset_noise=False):

    bsz = latents.shape[0]
    if noise is None:
        noise = torch.randn_like(latents)  

    if use_offset_noise:
        noise +=  0.1 * torch.randn(latents.shape[0], latents.shape[1], 1, 1).to(latents.device)
    # sample random timestep for each image
    if timesteps is None:
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (bsz,), device=latents.device,) 
        timesteps = timesteps.long()
    noisy_latent = scheduler.add_noise(latents, noise, timesteps)
    latents = scheduler.scale_model_input(latents, timesteps)

    return noisy_latent, noise, timesteps

def get_timesteps(num_inference_steps, strength, scheduler):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = scheduler.timesteps[t_start * scheduler.order :]

        return timesteps, num_inference_steps - t_start

class InpaintModel(pl.LightningModule):

    def __init__(self,
                 num_tokens=5,
                 use_offset_noise=True,
                 use_adapter=True,
                 disable_text_in_cond=False,
                 disable_ar_cond=False,
                 disable_geo = False,
                 train_kv=False,
                 opt_config=None,
                 ckpt_path=None,
                 sd_model_id=None,
                 clip_model_id=None,
                 transformer_config=None,
                 controlnet_id=None,
                 cache_dir=None):

        super().__init__() 
        self.cache_dir = cache_dir 
        self.sd_model_id = sd_model_id 
        self.clip_model_id = clip_model_id 
        self.opt_config = opt_config 
        self.train_kv = train_kv
        self.num_auxi_tokens = num_tokens
        self.use_offset_noise = use_offset_noise
        self.use_adapter = use_adapter
        self.disable_ar_cond = disable_ar_cond
        self.disable_text_in_cond = disable_text_in_cond
        self.disable_geo = disable_geo

        self.topil = T.ToPILImage()
        self.init_models(sd_model_id, clip_model_id, transformer_config, controlnet_id)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path) 




    def on_save_checkpoint(self, checkpoint) -> None:
        # custom checkpointing 
        saved_params = dict()
        for key, net in self.net_to_save.items():
            name_params = dict()
            for name, params in net.named_parameters():
                name_params[name] = params
            saved_params[key] = name_params
        checkpoint['state_dict'] = saved_params

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]

        for key, net in self.net_to_save.items():
            if key in self.no_grad_list:
                continue
            net.load_state_dict(sd[key], strict=False)
            print(f"loading {key}")
            if key not in self.no_grad_list: 
                net.requires_grad_(True)
                net.train()

        for key, net in self.net_to_freeze.items():
            net.requires_grad_(False)
        print(f"Restored from {path}")
    
    def init_models(self, sd_model_id, clip_model_id, transformer_config, controlnet_id=None):
        self.vae = AutoencoderKL.from_pretrained(sd_model_id, subfolder="vae", cache_dir=self.cache_dir)
        self.unet = UNet2DConditionModel.from_pretrained(sd_model_id, subfolder="unet", cache_dir=self.cache_dir)
        self.noise_scheduler = DDPMScheduler.from_pretrained(sd_model_id, subfolder='scheduler', cache_dir=self.cache_dir)
        
        try:
            self.image_processor = CLIPImageProcessor.from_pretrained(clip_model_id, subfolder="feature_extractor", cache_dir=self.cache_dir)
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_model_id, subfolder="image_encoder", cache_dir=self.cache_dir)
        except:
            self.image_processor = CLIPImageProcessor.from_pretrained(clip_model_id, cache_dir=self.cache_dir)
            self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(clip_model_id, cache_dir=self.cache_dir)
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_id, subfolder="tokenizer",cache_dir=self.cache_dir)
            self.text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_model_id, subfolder="text_encoder", cache_dir=self.cache_dir)
        except:
            self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_id, cache_dir=self.cache_dir)
            self.text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_model_id, cache_dir=self.cache_dir)

        clip_dim = self.image_encoder.config.projection_dim 


        self.uncond_embed = torch.nn.Embedding(77, self.text_encoder.config.hidden_size)
        self.net_to_save = dict()
        self.net_to_freeze = dict()
        self.uncond_embed.requires_grad_(True)

        self.net_to_save['uncond_embed'] = self.uncond_embed

        self.no_grad_list = list() 
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        self.net_to_freeze['vae'] = self.vae
        self.net_to_freeze['image_encoder'] = self.image_encoder
        self.net_to_freeze['text_encoder'] = self.text_encoder
        self.net_to_freeze['unet'] = self.unet

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)



        self.vae_image_processor = VaeImageProcessor(self.vae_scale_factor)
           

        if self.use_adapter:
            inchannel = 5
            if self.disable_geo:
                inchannel = 3
            self.adapter = T2IAdapter(in_channels=inchannel)
            adapter_transformer_config = transformer_config
            adapter_transformer_config['dim'] = self.adapter.channels[-1]
            self.adapter_transformer = CusTransformer(**transformer_config)
            self.adapter_transformer.requires_grad_(True)
            self.adapter.requires_grad_(True)
            self.net_to_save['adapter_transformer'] = self.adapter_transformer
            self.net_to_save['adapter'] = self.adapter

        # if self.train_kv:
        #     self.unet = create_custom_diffusion(self.unet)
        #     self.net_to_save['unet'] = self.unet
        #     del self.net_to_freeze['unet']

        if not self.disable_ar_cond:
            self.pos_embed = torch.nn.Embedding(200, self.text_encoder.config.hidden_size)
            transformer_config['dim'] = clip_dim
            self.projector = CusTransformer(**transformer_config)
            self.tgt = torch.nn.Embedding(self.num_auxi_tokens, self.text_encoder.config.hidden_size)
            self.adain = torch.nn.Sequential(
                FeedForward(self.text_encoder.config.hidden_size*2, 512, dropout=0.2),
                FeedForward(self.text_encoder.config.hidden_size*2, 512, dropout=0.2),
                torch.nn.Linear(self.text_encoder.config.hidden_size*2, self.text_encoder.config.hidden_size),
                FeedForward(self.text_encoder.config.hidden_size, 512, dropout=0.2),
                FeedForward(self.text_encoder.config.hidden_size, 512, dropout=0.2),
            )

            self.vision_embedding = torch.nn.Embedding(1, self.text_encoder.config.hidden_size)
            self.text_embedding = torch.nn.Embedding(1, self.text_encoder.config.hidden_size)


            self.tgt.requires_grad_(True)
            self.adain.requires_grad_(True)
            self.vision_embedding.requires_grad_(True)
            self.text_embedding.requires_grad_(True)
            self.pos_embed.requires_grad_(True)
            self.projector.requires_grad_(True)

            # this is the part of model we would like to train
            self.net_to_save['tgt'] = self.tgt
            self.net_to_save['projector'] = self.projector
            self.net_to_save['adain'] = self.adain
            self.net_to_save['text_embedding'] = self.text_embedding
            self.net_to_save['vision_embedding'] = self.vision_embedding
            self.net_to_save['pos_embed'] = self.pos_embed





            

    @torch.no_grad()
    def encode_image(self, image, generator=None):
        latents = self.vae.encode(image).latent_dist.sample(generator)
        latents = latents * self.vae.config.scaling_factor
        return latents 
    
        
    @torch.no_grad()
    def decode_latent(self, latents, rescale=False):
        
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = image.clamp(-1, 1)
        if rescale:
            # scale to [0, 1]
            image += 1
            image /= 2
        # image = (image / 2 + 0.5).clamp(0, 1)
        return image

    # @torch.no_grad()
    def clip_encode_prompt(self, prompts, pool=False, padding='max_length', extra_embedding=None):
        prompts_ids = self.tokenizer(prompts, padding=padding, max_length=self.tokenizer.model_max_length, truncation=True,return_tensors='pt').input_ids
        
        output = self.text_encoder(input_ids=prompts_ids.to(self.device), extra_embedding=extra_embedding)

        if pool:
            prompts_emb = output.text_embeds
        else:
            prompts_emb = output.last_hidden_state


        return prompts_emb

        
        
    
    @torch.no_grad()
    def clip_encode_image(self, image, add_null_head=True, pool=True, remove_cls=True):
        # make sure image in range -1 to 1
        image = (image + 1)/2
        processed_list = list()
        for img in image:
            img = self.topil(img)
            img = self.image_processor(images=img, return_tensors='pt').pixel_values.to(image.device)
            processed_list.append(img)

        image = torch.stack(processed_list, dim=0).squeeze(dim=1)
        if pool:
            clip_feat = self.image_encoder(pixel_values=image).image_embeds
        else:
            clip_feat = self.image_encoder(pixel_values=image).last_hidden_state
            # remove class embed 
            if remove_cls:
                clip_feat = clip_feat[:, 1:]

            clip_feat = self.image_encoder.visual_projection(clip_feat)


        if not add_null_head:
            return clip_feat

        null_text_embed = self.clip_encode_prompt([""], padding='do_not_pad')
        null_head = null_text_embed[: ,0].unsqueeze(dim=1)
        null_tail = null_text_embed[: ,1].unsqueeze(dim=1)
        bzs = image.size(0)
        clip_feat = torch.cat([null_head.repeat(bzs, 1, 1).to(clip_feat.device), clip_feat, null_tail.repeat(bzs, 1, 1).to(clip_feat.device)], dim=1)
        return clip_feat




    def unet_pred(self, noisy_latents, timesteps, cond_embed, adapter_feat=None):

        

        model_pred = self.unet(
                            noisy_latents,
                            timesteps,
                            encoder_hidden_states=cond_embed,
                            down_block_additional_residuals=adapter_feat,
                            return_dict=False,
        )[0]
        return model_pred
    


    '''
    this is the branch to extract global feature
    '''
    def get_cond_embed(self, pixel_cube, image, caption, log_attn_map=False):

        if self.disable_ar_cond:
            cond_embed = self.clip_encode_prompt(caption)
            return cond_embed

        bs = pixel_cube.size(0) 
        n_patch = pixel_cube.size(1) 

        # global vision in a form of cubemap 
        cube_feat = rearrange(pixel_cube, 'b p c h w -> (b p) c h w')
        cube_feat = self.clip_encode_image(cube_feat, False, True, False)
        cube_feat = rearrange(cube_feat, '(b p) c -> b p c', b=bs, p=n_patch)

        # the nfov image
        input_feat = self.clip_encode_image(image, False, True, False)
        input_feat = rearrange(input_feat, 'b c -> b 1 c')

        # text 
        text_feat = self.clip_encode_prompt(caption)

        # use embedding to distinguish the vision and text feature
        text_embedding = self.text_embedding.weight
        text_embedding = rearrange(text_embedding, 'n c -> 1 n c').repeat(bs, 1, 1)
        vision_embedding = self.vision_embedding.weight
        vision_embedding = rearrange(vision_embedding, 'n c -> 1 n c').repeat(bs, 1, 1)
        mem = torch.cat([cube_feat+vision_embedding, text_feat+text_embedding], dim=1)


        # for ablation
        if self.disable_text_in_cond:
            mem = cube_feat+vision_embedding

        # use current patch to query the global vision + text 
        tgt = self.tgt.weight
        n = tgt.size(0)
        tgt = rearrange(tgt, 'n c -> 1 n c').repeat(bs, 1, 1) 
        input_feat = input_feat.repeat(1, n, 1)
        tgt = torch.cat([tgt, input_feat], dim=-1)
        tgt = self.adain(tgt)

        mem = torch.cat([tgt, mem], dim=1)
        k = mem.size(1)

        pos_embed = self.pos_embed(torch.LongTensor(list(range(k))).to(self.device))
        pos_embed = rearrange(pos_embed, 'n c -> 1 n c').repeat(bs, 1, 1)

        mem = mem + pos_embed

    
        y = self.projector(mem, log_attn_map=log_attn_map)
        y = y[:, :n]

        # get the extracted feature with text prompt
        cond_embed = self.clip_encode_prompt(caption)
        cond_embed = torch.cat([y, cond_embed], dim=1)

        return cond_embed

    def get_uncond_embed(self, bs):
        uncond_embed = self.uncond_embed(torch.LongTensor(list(range(5))).to(self.device))
        uncond_embed = rearrange(uncond_embed, 'n c -> 1 n c').repeat(bs, 1, 1)
        return uncond_embed


    '''
        this is the branch to extract local feature
    '''
    def get_adapter_feat(self, pixel_cube, pos_cube, image, pos):
        if not self.use_adapter:
            return None
        cube = torch.cat([pixel_cube, pos_cube], dim=2)
        pers = torch.cat([image, pos], dim=1)

        if self.disable_geo:
            cube = pixel_cube
            pers = image


        b, n, c, h, w = cube.shape
        cube = rearrange(cube, 'b n c h w -> (b n) c h w')

        cube_feats = self.adapter(cube) 
        pers_feats = self.adapter(pers)
        adapter_feat = pers_feats

        b, c, h, w = pers_feats[-1].shape

        src = rearrange(pers_feats[-1], "b c h w -> b (h w) c")
        tgt = rearrange(cube_feats[-1], '(b n) c h w -> b (n h w) c', b=b, n=n)
        middle_feat = self.adapter_transformer(src, tgt)
        
        middle_feat = rearrange(middle_feat, 'b (h w) c -> b c h w', h=h, w=w)

        adapter_feat[-1] = middle_feat

        return adapter_feat





    @torch.no_grad()
    def inpaint(self,
                image,
                mask_image,
                prompt=[""],
                prompt_embeds=None,
                num_inference_steps=50,
                guidance_scale=2.5,
                strength=1.,
                generator=None,
                scheduler=None,
                adapter_feat=None,
                learned_uncond=True):

        scheduler = self.noise_scheduler if scheduler is None else scheduler 
        device = self.device

        if prompt_embeds is None:
            prompt_embeds = self.clip_encode_prompt(prompt)
        if learned_uncond:
            uncond_prompt_embeds = self.get_uncond_embed(image.size(0))
        else:
            uncond_prompt_embeds = self.clip_encode_prompt([""]*image.size(0))

        # 4. set timesteps
        scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, num_inference_steps = get_timesteps(
            num_inference_steps=num_inference_steps, strength=strength,  scheduler=scheduler,
        )
        

        num_unet_channel = self.unet.config.in_channels

        # prepare inputs
        latents_size = image.size(-1) // 8
        mask, masked_image = prepare_mask_and_masked_image(image, mask_image)
        latents = randn_tensor((image.size(0), 4, latents_size, latents_size), device=self.device, generator=generator, dtype=prompt_embeds.dtype) * scheduler.init_noise_sigma
        noise = randn_tensor((image.size(0), 4, latents_size, latents_size), device=self.device, generator=generator, dtype=prompt_embeds.dtype) 
        masked_latents = self.encode_image(masked_image, generator=generator) 
        h = w = masked_latents.shape[-1]
        mask = torch.nn.functional.interpolate(mask, (h, w)) 


        # denoiseing step
        for i, t in enumerate(timesteps):
            scaled_latents = scheduler.scale_model_input(latents, t)

            if num_unet_channel == 9:
                latent_model_input = torch.cat([scaled_latents, mask, masked_latents], dim=1)
            else:
                latent_model_input = scaled_latents
            
            cond_noise = self.unet_pred(latent_model_input, t, prompt_embeds, adapter_feat)
            uncond_noise = self.unet_pred(latent_model_input, t, uncond_prompt_embeds, adapter_feat)
            noise_pred = uncond_noise + guidance_scale * (cond_noise - uncond_noise)

            latents = scheduler.step(noise_pred, t, latents, ).prev_sample

            if num_unet_channel == 4:
                init_latents_proper = masked_latents
                init_mask = mask

                if i < len(timesteps) - 1:
                    noise_timestep = timesteps[i + 1]
                    init_latents_proper = scheduler.add_noise(
                        init_latents_proper, noise, torch.tensor([noise_timestep])
                    )

                latents = (1 - init_mask) * init_latents_proper + init_mask * latents
            
        output_image = self.decode_latent(latents).detach().cpu()
        ret = dict() 
        ret['output'] = output_image
        ret['input'] = image 
        ret['masked_input'] = masked_image


        return ret

        return None
    def forward(self, batch):
        pixel_cube, pos_cube, image, mask, target, caption, pos = batch

        b, n, c, h ,w = pixel_cube.shape

        # 1. encode image to latent
        latent = self.encode_image(target)
        noisy_latent, noise, timesteps = add_noise(latent, self.noise_scheduler, use_offset_noise=self.use_offset_noise)

        # 1.1 global_cond_embed
        if random.uniform(0, 1) < 0.8:
            cond_embed = self.get_cond_embed(pixel_cube, image, caption,)
            
        else:
            cond_embed = self.get_uncond_embed(latent.size(0)).to(self.device)

        # 1.2 local feat
        adapter_feat = None
        if self.use_adapter:
            adapter_feat = self.get_adapter_feat(pixel_cube, pos_cube, image, pos)


        # 2. encode masked_image 
        masked_latent =  self.encode_image(image * (1-mask))

        # 3 reshape mask to latent shape
        h, w = masked_latent.size(-2), masked_latent.size(-1)
        mask = torch.nn.functional.interpolate(mask, (h, w))

        if self.unet.config.in_channels == 9:
            noisy_latent = torch.cat([noisy_latent, mask, masked_latent,], dim=1)


        if self.noise_scheduler.config.prediction_type == 'epsilon':
            target = noise
        elif self.noise_scheduler.config.prediction_type == 'v_prediction':
            target = self.noise_scheduler.get_velocity(latent, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

        model_pred = self.unet_pred(noisy_latent, timesteps, cond_embed, adapter_feat)
        loss = torch.nn.functional.mse_loss(model_pred.float(), target.float())

        return loss



    def training_step(self, batch, batch_ids):
        loss = self.forward(batch)
        self.log('train/loss', loss.detach().item(), logger=True, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_ids):
        loss = self.forward(batch)
        self.log('val/loss', loss.detach().item(), logger=True, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):

        params = list() 
        for name, net in self.net_to_save.items():
            if name in self.no_grad_list:
                print(f"skipping {name}")
            else:
                print(name, " loading param to train")
                params += net.parameters()

        opt = torch.optim.AdamW(params, **self.opt_config)
        return opt
    

    @torch.no_grad()
    def log_images(self, batch, inference_step=50, guidance=2.5, resize_images=True):

        scheduler = DDIMScheduler.from_pretrained(self.sd_model_id, subfolder='scheduler', cache_dir=self.cache_dir)
        pixel_cube, pos_cube, image, mask, target, caption, pos = batch
        
        # local + global feat
        cond_embed = self.get_cond_embed(pixel_cube, image, caption,)
        adapter_feat = None
        if self.use_adapter:
            adapter_feat = self.get_adapter_feat(pixel_cube, pos_cube, image, pos)
        ret = self.inpaint(image, mask, prompt=caption, prompt_embeds=cond_embed, num_inference_steps=inference_step, guidance_scale=guidance, scheduler=scheduler,adapter_feat=adapter_feat )
        ret['cond_image'] = image
        ret['gt'] = target
    

        for k, v in ret.items():
            ret[k] = v.detach().cpu()
        return ret, target.size(0), caption

    
    



