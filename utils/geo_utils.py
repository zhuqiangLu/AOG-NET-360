import random
from equilib.equi2pers.numpy import *
import numpy as np
import torch 
import equilib
from PIL import Image

from .pers2equi import pers2equi
from equilib.equi2pers.torch import create_rotation_matrices
from equilib.equi2cube.torch import create_xyz_grid

mask_indicator = -100
def random_rot(vertical_angle: int=180, horizontal_angle: int=360):
    rots = {
        'roll': 0.,
        'pitch': np.radians(random.randint(0, vertical_angle)),
        'yaw': np.radians(random.randint(0, horizontal_angle))
    }

    return rots


def get_pos(cube_size, z_down=False, rots={'roll': 0, 'yaw': 0, 'pitch': 0}):

    xyz = create_xyz_grid(w_face=cube_size, batch=1, dtype=torch.float32) 

    xyz = xyz.unsqueeze(-1)

    z_down = not z_down
    R = create_rotation_matrices([rots], z_down)

    M = torch.matmul(R[:, None, None, ...], xyz)
    xyz = M.squeeze(-1)
    phi = torch.asin(xyz[..., 2] / torch.norm(xyz, dim=-1)) # -90 to 90
    theta = torch.atan2(xyz[..., 1], xyz[..., 0]) # -180 to 180
    # print(phi.min(), phi.max(), theta.min(), theta.max())
    
    pos = torch.stack([phi, theta], dim=-1).squeeze(dim=0)
    cube_list = list(torch.split(pos, split_size_or_sections=pos.shape[0], dim=1))
    cube_dict = dict()
    k = ["F", "R", "B", "L", "U", "D"]
    for ke, cube in zip(k, cube_list):
        cube_dict[ke] = cube.permute(-1, 0, 1) # / torch.pi
    
    return cube_dict


def put_back(gen_equi, patch, yaw, pitch):
    
    c, h, w = gen_equi.shape
    equi, mask = pers2equi(patch, yaw, pitch, w, h) 
    indicator_mask = (gen_equi == mask_indicator) * 1.
    mask = mask * indicator_mask 
    gen_equi = torch.where(mask==1., equi, gen_equi)
    
    return gen_equi


def get_nfov(image, yaw, pitch, height=512, width=512, fov_x=90, mode='nearest'):
    # image shape : (c, h, w)
    pitch = (pitch / 180 * np.pi)
    yaw = (yaw / 180 * np.pi)
    rots={'roll': 0, 'pitch': pitch, 'yaw': yaw}
    if len(image.shape) == 4 and not isinstance(rots, list):
        rots = [rots]
    nfov = equilib.equi2pers(image, rots=rots, height=height, width=width, fov_x=fov_x, z_down=False, mode=mode)
    return nfov, rots


def save(*imgs, dest, cast=True):
    ret = list() 
    for tmp_img in imgs:
        img = tmp_img.clone()
        
        if cast:
            if isinstance(img, torch.Tensor):
                img = img.cpu().detach().numpy()
            if img.min() < 0:
                img += 1
                img /= 2
            img = (img * 255).astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))
        ret.append(img) 
    img = np.concatenate(ret, axis=0)
    img = Image.fromarray(img) 
    print(f'saving to {dest}')
    img.save(dest)

def show(*imgs, cast=True):
    
    ret = list()
    for temp in imgs:
        img = temp.clone()
        if isinstance(img, torch.Tensor):
            img = img.cpu().detach().numpy()
        if img.min() < 0:
            img += 1
            img /= 2
        if cast:
            # img = (img * 0.5) + 0.5
            img = (img * 255).astype(np.uint8)
        img = np.transpose(img, (1, 2, 0))
        ret.append(img)
    img = np.concatenate(ret, axis=0)
    img = Image.fromarray(img)
    img.show()

        



   





    









 
