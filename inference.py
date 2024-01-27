import torch 
from utils.geo_utils import save, mask_indicator, get_nfov, put_back
import equilib
from einops import rearrange
from dataset import  gen_coord_list
import gradio as gr
from tqdm import tqdm
from utils.config_utils import get_configs, instantiate_from_config
from diffusers import PNDMScheduler
from omegaconf import OmegaConf
import os
import torchvision.transforms as T 
from rich.progress import track



parser, config = get_configs() 
topil = T.ToPILImage()
device='cuda'
i = 0
nfov_size = 512
out_image_height = 2048
out_image_width = 4096

dl_config = config.get('data', OmegaConf.create())
dl = instantiate_from_config(dl_config)
dl.setup()
ds = dl.val_dataloader().dataset

equi_pos = ds.equi_pos
pos = equilib.equi2cube(ds.equi_pos, w_face=ds.cube_size, rots={'yaw': 0, 'pitch': 0, 'roll':0}, cube_format='dict')
gt_equi = ds.get_gt_equi(i,) 
global_caption = ds.get_caption(i)


model_config = config.get('model', OmegaConf.create())
model = instantiate_from_config(model_config)

name = parser.name

ckpt = parser.checkpoint
model.init_from_ckpt(ckpt)
model = model.to(device)


num_inference_steps = parser.num_inference_step
guidance_scale = parser.cfg_scale 
scheduler = PNDMScheduler.from_pretrained(model.sd_model_id, subfolder='scheduler', cache_dir=model.cache_dir)

counter = 0
global_cond_embed = list() 




def get_input(target_yaw, target_pitch, gen_equi_local):
    masked_equi = torch.where(gen_equi_local==mask_indicator, -1, gen_equi_local)
    yaw = target_yaw
    pitch = target_pitch
    equi_mask = torch.where(gen_equi_local==mask_indicator, 1, 0)[:1].to(torch.float32)

    pixel_cube = ds.get_cube(masked_equi) 
    pos_cube = ds.get_cube(ds.equi_pos) 


    input_image, _ = get_nfov(masked_equi, yaw, pitch, ds.cube_size, ds.cube_size,)
    mask, _ = get_nfov(equi_mask, yaw, pitch, ds.cube_size, ds.cube_size)
    target, _ = get_nfov(gt_equi, yaw, pitch, ds.cube_size, ds.cube_size)
    pos, _ = get_nfov(ds.equi_pos, yaw, pitch, ds.cube_size, ds.cube_size)

    input_image = rearrange(input_image, 'c h w -> 1 c h w').to(device)
    target = rearrange(target, 'c h w -> 1 c h w').to(device)
    mask = rearrange(mask, 'c h w -> 1 c h w').to(device)
    pos = rearrange(pos, 'c h w -> 1 c h w').to(device)
    pixel_cube = rearrange(pixel_cube, 'n c h w -> 1 n c h w').to(device)
    pos_cube = rearrange(pos_cube, 'n c h w -> 1 n c h w').to(device)
    caption = global_caption

    return pixel_cube, pos_cube, input_image, mask, target, caption, pos


def inpaint_all():

    dest = parser.inference_out_dir
    exp_name = parser.name
    dest = os.path.join(dest, exp_name)
    os.makedirs(dest, exist_ok=True)

    for i in track(range(len(ds))):
        gt_equi_local = ds.get_gt_equi(i) 
        source_image_local, _ = get_nfov(gt_equi_local, 0, 0, height=nfov_size, width=nfov_size, fov_x=90)
        gen_equi_local = torch.ones((3, out_image_height, out_image_width),) * mask_indicator
        
        gen_equi_local = put_back(gen_equi_local, source_image_local, 0, 0)
        mask = torch.ones_like(gen_equi_local) 
        input_equi = gen_equi_local


        cur_dest = os.path.join(dest, str(i))
        os.makedirs(cur_dest, exist_ok=True)

        
        for jdx, (yaw, pitch) in tqdm(enumerate(gen_coord_list)):
            gt_target, _ = get_nfov(gt_equi_local, yaw, pitch)
            gt_target = rearrange(gt_target, 'c h w -> 1 c h w')

            pixel_cube, pos_cube, input_image, mask, target, caption, pos = get_input(yaw, pitch, gen_equi_local)
            adapter_feat = model.get_adapter_feat(pixel_cube, pos_cube, input_image, pos)
            prompt_embed = model.get_cond_embed(pixel_cube, input_image,  caption)
            ret = model.inpaint(input_image, 
                                      mask, 
                                      prompt_embeds=prompt_embed,
                                      prompt=caption,
                                      num_inference_steps=num_inference_steps,
                                      guidance_scale=guidance_scale,
                                      adapter_feat=adapter_feat,
                                      scheduler=scheduler)
            ret_image = ret['output'][0]
            gen_equi_local = put_back(gen_equi_local.clone(), ret_image.cpu().detach().clone(), yaw, pitch)
            

        save(gen_equi_local, dest=os.path.join(cur_dest, 'result.png'), cast=True)
        save(gt_equi_local, dest=os.path.join(cur_dest, 'gt.png'), cast=True)
        save(input_equi, dest=os.path.join(cur_dest, 'input.png'), cast=True)
        path = ds.paths[i]
        with open(os.path.join(cur_dest, 'path.text'), 'w') as f:
            f.write(str(path))

        with open(os.path.join(cur_dest, 'prompt.text'), 'w') as f:
            f.write(caption)
            





if __name__ == "__main__":
    inpaint_all()







    
   
    
