from torch.utils.data import Dataset
import os
from equilib import equi2cube
from pathlib import Path
import torchvision.transforms as T
from PIL import Image
import random 
import torch 
from utils.geo_utils import get_nfov, show, save, random_rot, get_pos, put_back, mask_indicator
import pytorch_lightning as pl
import torch.utils.data as data
from einops import rearrange
from utils.config_utils import instantiate_from_config
import torch.utils.data as data
import pickle
import equilib
import numpy as np
from textattack.augmentation import EmbeddingAugmenter


gen_coord_list = [
        (0, -25,), (0,25), 
        (-45, 0),# (-45, -25), (-45, 25),
        (45, 0), #(45, -25), (45, 25),
        (-90, 0), (-90, -25), (-90, 25),
        (90, 0), (90, -25), (90, 25),
        (-105, 0), (105, 0), (180, 0), (180, 25), (180, -25), 
        (0, -45), (90, -45),  (-90, -45,), (0, 45), (-90, 45), (90, 45), ] # 21

# gen_coord_list = [
#         (0, -25,), (0,25), 
#         (-30, 0), (-60, 0),
#         (30, 0), (60, 0),
#         (-90, 0), (-90, -25), (-90, 25),
#         (90, 0), (90, -25), (90, 25),
#         (-125, 0), (125, 0), (180, 0), (180, 25), (180, -25), 
#         (0, -45), (90, -45),  (-90, -45,), (0, 45), (-90, 45), (90, 45), ] # 23
# 
# gen_coord_list = [
#         (0, -25), (0, 25),
#         (-75, 0),(-75, -25), (-75, 25),
#         (75, 0),(75, -25), (75, 25),
#         (-150, 0), (-150, -25), (-150, 25),
#         (150, 0), (150, -25), (150, 25),
#         (180, 0), 
#         (0, -45), (90, -45),  (-90, -45,), (0, 45), (-90, 45), (90, 45), ]




# gen_coord_list_60 = [
#         (0, -25,), (0,25), 
#         (-60, 0),(-60, -25), (-60, 25),
#         (60, 0), (60, -25), (60, 25),
#         (-105, 0), (-105, -25), (-105, 25),
#         (105, 0), (105, -25), (105, 25),
#         (180, 0), (180, 25), (180, -25),
#         (0, -45), (90, -45),  (-90, -45,), (0, 45), (-90, 45), (90, 45), ]

# gen_coord_list_15= [
# 
#         (0, -25,), (0,25), 
#         (-90, 0), (-90, -25), (-90, 25),
#         (90, 0), (90, -25), (90, 25),
#         (180, 0), (180, -25), (180, 25), 
#         (0, -45), (90, -45),  (-90, -45,), (0, 45), (-90, 45), (90, 45), ]
class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 wrap=False, num_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        

    def _train_dataloader(self):
        return data.DataLoader(self.datasets["train"], batch_size=self.batch_size,
        num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def _val_dataloader(self):
        return data.DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True, pin_memory=True)

    def _test_dataloader(self):
        return data.DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True)

class ImageDataset(Dataset):
    def __init__(self, folders, image_width, image_height, cube_size, image_scale=1., dataset_size=-1, folders_size=None, no_360_aug=False, exts=['png', 'jpg', 'jpeg'], downsample=False):
        super().__init__()
        self.image_scale = image_scale
        self.image_width = image_width
        self.image_height = image_height
        self.downsample = downsample
        self.no_360_aug = no_360_aug

        if not isinstance(folders, list):
            folders = list(folders) 

        if not isinstance(exts, list):
            exts = list(exts)

        if folders_size is None:
            folders_size = [-1 for _ in range(len(folders))]
        else:
            folders_size = list(folders_size)
        
        self.paths = list()
        for folder, folder_size in zip(folders, folders_size):
            images = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
            # random.shuffle(images)
            if folder_size == -1:
                self.paths += images
                print(f'from {folder} import {len(images)} images')
            else:
                self.paths += images[:folder_size]
                print(f'from {folder} import {len(images[:folder_size])} images')

        if dataset_size > 0 and dataset_size < len(self.paths):
            self.paths = random.sample(self.paths, dataset_size)
            print(f"randomly sample {len(self.paths)} sample from all dataset")


        
        self.transfomer = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize([image_height, image_width]),
            T.ToTensor()
        ])

        self.topil = T.ToPILImage()

        self.augment = T.Compose(
            [
                T.ToPILImage(),
                T.RandomCrop((cube_size, cube_size)),
                T.ToTensor(),
            ]
        )

        self.augment2 = T.Compose(
            [
                T.ToTensor(),
            ]
        )
        self.cube_size = cube_size


       

    def __len__(self):
        
        return len(self.paths) 
    

    def __getitem__(self, idx):

        path = self.paths[idx]    
        img = Image.open(path)
        img = self.transfomer(img) # c h w
        # rots = self.get_rots()
        aug_deg = 360
        if self.no_360_aug:
            aug_deg = 0 
        rots = random_rot(vertical_angle=0, horizontal_angle=aug_deg)
     

        # equi augmentation 
        equi = equilib.equi2equi(img, rots, z_down=False, mode='bilinear',)

        target_image, target_mask, masked_target, _ = self.sampler.get_training_sample(equi, None)
        target_mask = target_mask * 1.
        target_mask = target_mask.to(target_image.dtype)
        
        target_image = self.augment(target_image) * 2 - 1
        return target_image, target_mask



    def get_all_faces(self, idx, vertical_angle_range=0, horizontal_angle_range=0):
        path = self.paths[idx]    
        img = Image.open(path)
        img = self.transfomer(img) # c h w
        # rots = self.get_rots()
        rots = random_rot(vertical_angle=vertical_angle_range, horizontal_angle=horizontal_angle_range)

        pos = get_pos(rots=rots, z_down=False, cube_size=self.cube_size)


        cubes = equi2cube(img, rots, 
                          w_face=self.cube_size, 
                          cube_format='dict',
                          z_down=False)

        faces = dict() 
        keys = list(cubes.keys())
        for i, key in enumerate(keys):
            faces[key] = cubes[key] #* 2 - 1


        inference_faces = dict()
        for i, key in enumerate(keys):
            # inference_faces[key] = self.augment2(cubes[key])
            if key == 'F':
                inference_faces[key] = cubes[key] # * 2 - 1
            else:
                inference_faces[key] = torch.zeros_like(cubes[key]) #* 2 - 1


        return faces, inference_faces, pos
    

    def get_gt_equi(self, idx, width=2048, height=1024):
        
        path = self.paths[idx]
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img = img.resize((width, height)) 
        img = np.array(img).astype(np.float32) / 255. 
        img = torch.from_numpy(img) 
        img = rearrange(img, 'h w c -> c h w')

        return img
        


class PosImageDataset(ImageDataset):

    def __init__(self, caption_info=None, mask_info=None, augment_caption_info=None, num_augment_caption=5, wordswap=True, *args, **kwargs):
        super(PosImageDataset, self).__init__(*args, **kwargs)
        self.pos = get_pos(rots={'yaw': 0, 'pitch': 0, 'roll':0}, z_down=False, cube_size=2048)
        self.equi_pos = equilib.cube2equi(self.pos, width=self.image_width, height=self.image_height, cube_format='dict', mode='nearest')
        self.mask_indicator = mask_indicator
        self.gen_coord_list = gen_coord_list
        self.get_equi_mask(mask_info)
            
        self.captions = None
        self.augment_captions = None
        self.num_augment_caption = num_augment_caption


        # load the pre compute image captions 
        if caption_info is not None:
            with open(caption_info, 'rb') as f:
                self.captions = pickle.load(f)
                print(f"loading {len(self.captions)} captions")

        if augment_caption_info is not None:
            with open(augment_caption_info, 'rb') as f:
                self.augment_captions = pickle.load(f)

        
        self.nlp_augmenter = EmbeddingAugmenter() if wordswap else None
        




    def set_mask_idx(self, idx=0):
        self.mask_idx = idx

    def get_gt_equi(self, idx,):
        path = self.paths[idx]    
        img = Image.open(path)
        img = self.transfomer(img) # c h w , in [0, 1]
        img = img * 2 - 1
        return img


    def __getitem__(self, idx):

        path = self.paths[idx]    
        caption = self.get_caption(idx)
        img = Image.open(path)
        img = self.transfomer(img) # c h w , in [0, 1]
        img = img * 2 - 1

        rots = random_rot(vertical_angle=0, horizontal_angle=360)

        # equi augmentation 
        equi = equilib.equi2equi(img, rots, z_down=False, mode='bilinear',)
        equi = torch.clamp(equi, -1, 1)

        step_idx = random.choice(list(range(len(gen_coord_list)-3)))
        yaw, pitch = gen_coord_list[step_idx]
        equi_mask = self.equi_mask[step_idx][:1] 
        masked_equi = torch.where(equi_mask, -1, equi)

        pixel_cube = self.get_cube(masked_equi) 
        pos_cube = self.get_cube(self.equi_pos) 


        input, _ = get_nfov(masked_equi, yaw, pitch, self.cube_size, self.cube_size,)
        mask, _ = get_nfov(equi_mask*1., yaw, pitch, self.cube_size, self.cube_size)
        pos, _ = get_nfov(self.equi_pos, yaw, pitch, self.cube_size, self.cube_size)
        target, _ = get_nfov(equi, yaw, pitch, self.cube_size, self.cube_size)

        
        return pixel_cube, pos_cube, input, mask, target, caption, pos


    
    
    def get_patch(self, equi):
        # remove the last three as it is a stand 
        step_idx = random.choice(list(range(len(gen_coord_list)-3)))
        equi_mask = self.equi_mask[step_idx][:1] 
        # masked_equi = torch.where(equi_mask, self.mask_indicator, equi)


        img = torch.cat([equi, equi_mask, self.equi_pos])

        local_patch = list() 
        local_mask = list() 
        local_pos = list()

        for (yaw, pitch) in self.gen_coord_list:
            patch, _ = get_nfov(img, yaw, pitch, self.cube_size, self.cube_size, mode='nearest')
            gt_target = patch[:3]
            mask = patch[3:4]
            target_pos = patch[4:]

            local_patch.append(gt_target)
            local_mask.append(mask)
            local_pos.append(target_pos)

        local_patch = torch.stack(local_patch, dim=0)
        local_mask = torch.stack(local_mask, dim=0) * 1.
        local_pos = torch.stack(local_pos, dim=0)

        return local_patch, local_mask, local_pos, equi_mask, self.gen_coord_list[step_idx]


    def get_cube(self, masked_equi):
        rots = {
            'roll': 0., 
            'pitch': 0,
            'yaw': 0., 
        }

        cube = equilib.equi2cube(masked_equi, rots=rots, w_face=self.cube_size, cube_format='dict', mode='nearest')

        cube_list = list()
        for k, v in cube.items():
            cube_list.append(v) 

        cube = torch.stack(cube_list, dim=0)
        return cube







    def get_equi_mask(self, mask_info_path):

        # if mask_info_path is None:
        if not os.path.isfile(mask_info_path):
            # create a mask for each coords
            # note that coords stand for target coords, therefore there is no (0, 0)

            print('init mask for faster training')
            coords = [(0, 0)] + gen_coord_list 
            mask_list = list() 
            gen_equi = torch.ones(3, self.image_height, self.image_width) * self.mask_indicator
            one_patch = torch.ones(3, self.cube_size, self.cube_size)
            for yaw, pitch in coords:
                gen_equi = put_back(gen_equi, one_patch, yaw, pitch) 
                gen_equi_mask = (gen_equi == self.mask_indicator) 
                mask_list.append(gen_equi_mask)

            self.equi_mask = mask_list[:-1]
            # save cache 
            with open(mask_info_path, 'wb') as f:
                save_dict = dict() 
                save_dict['coords'] = self.gen_coord_list
                save_dict['mask_indicator'] = self.mask_indicator
                save_dict['mask'] = self.equi_mask
                pickle.dump(save_dict, f)
            
        else:
            with open(mask_info_path, 'rb') as f:
                inpaint_scheme = pickle.load(f)
                self.gen_coord_list = inpaint_scheme['coords'] 
                self.equi_mask = inpaint_scheme['mask']
                self.mask_indicator = inpaint_scheme['mask_indicator']
                
        
    def get_caption(self, idx):

        if self.captions is None:
            return ""
        path = self.paths[idx]    
        file_name = str(path).split('/')[-1] 
        caption = self.captions[file_name][0]
        
        num_augment_sample = 0
        if self.augment_captions is not None:
            if self.num_augment_caption != 0:
                num_augment_sample = random.choice(range(min(self.num_augment_caption, len(self.augment_captions[file_name]))))

            augment_captions = random.sample(self.augment_captions[file_name], num_augment_sample)
            caption = ", ".join([caption] + augment_captions)

        if self.nlp_augmenter:
            caption = self.nlp_augmenter.augment(caption)[0]

            
        return caption






if __name__ == '__main__':
    from dataset import ImageDataset
    split = 'test'
    ds = PosImageDataset(caption_info=f"./image_caption/{split}_new.pkl", 
                         mask_info="./mask.pkl", 
                         augment_caption_info=f"./image_caption/{split}_augment.pkl",
                         num_augment_caption=10,
                         folders=[f'/home/zhlu6105/Datasets/laval/{split}/'], 
                         image_width=7168, 
                         image_height=3584, 
                         cube_size=512)


