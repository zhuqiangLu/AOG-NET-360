import torch 
import numpy as np 
import cv2
from kornia.geometry.transform import remap
from PIL import Image

import os
import sys
import cv2
import numpy as np


'''
This is the back projection method, nfov -> equiretangular 
'''
def pers2equi(imgs, yaw, pitch, equi_width=2048, equi_height=1024, fov=90):

    imgs = torch.flip(imgs, [1])
    theta = -yaw 
    phi = pitch

    _, img_height, img_width, = imgs.shape


    # prepare sampling matrices 
    wFOV = fov 
    hFOV = float(img_height)/img_width * fov

    w_len = np.tan(np.radians(wFOV/2.0))
    h_len = np.tan(np.radians(hFOV/2.0))

    x, y = np.meshgrid(np.linspace(-180, 180, equi_width), np.linspace(-90, 90, equi_height))
    x_map = np.cos(np.radians(x)) * np.cos(np.radians(y))
    y_map = np.sin(np.radians(x)) * np.cos(np.radians(y))
    z_map = np.sin(np.radians(y))

    xyz = np.stack((x_map,y_map,z_map),axis=2)

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)
    [R1, _] = cv2.Rodrigues(z_axis * np.radians(theta))
    [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-phi))

    R1 = np.linalg.inv(R1)
    R2 = np.linalg.inv(R2)

    xyz = xyz.reshape([equi_height * equi_width, 3]).T
    xyz = np.dot(R2, xyz)
    xyz = np.dot(R1, xyz).T

    xyz = xyz.reshape([equi_height , equi_width, 3])
    inverse_mask = np.where(xyz[:,:,0]>0,1,0)

    xyz[:,:] = xyz[:,:]/np.repeat(xyz[:,:,0][:, :, np.newaxis], 3, axis=2)
    
    
    lon_map = np.where((-w_len<xyz[:,:,1])&(xyz[:,:,1]<w_len)&(-h_len<xyz[:,:,2])
                &(xyz[:,:,2]<h_len),(xyz[:,:,1]+w_len)/2/w_len*img_width,0)
    lat_map = np.where((-w_len<xyz[:,:,1])&(xyz[:,:,1]<w_len)&(-h_len<xyz[:,:,2])
                    &(xyz[:,:,2]<h_len),(-xyz[:,:,2]+h_len)/2/h_len*img_height,0)
    mask = np.where((-w_len<xyz[:,:,1])&(xyz[:,:,1]<w_len)&(-h_len<xyz[:,:,2])
                    &(xyz[:,:,2]<h_len),1,0)

    mask = torch.from_numpy(mask)
    lon_map = torch.from_numpy(lon_map.astype(np.float32))
    lat_map = torch.from_numpy(lat_map.astype(np.float32))

    mask = mask * inverse_mask
    mask = np.repeat(mask[np.newaxis, :, :, ], 1, axis=0)
    
    # pers = remap(imgs.unsqueeze(dim=0), lon_map.unsqueeze(dim=0), lat_map.unsqueeze(dim=0), mode='nearest', padding_mode='reflection').squeeze(dim=0)
    pers = remap(imgs.unsqueeze(dim=0), lon_map.unsqueeze(dim=0), lat_map.unsqueeze(dim=0), mode='bilinear', padding_mode='reflection').squeeze(dim=0)
    # pers = pers * mask
    # pers = torch.flip(pers, [1, 2]) 
    # mask = torch.flip(mask, [1, 2]) 

    return pers, mask



