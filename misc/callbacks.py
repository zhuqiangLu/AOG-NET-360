import numpy as np
import pytorch_lightning
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only
import random
import torchvision

class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True, selected_keys=None, image_size=512):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images if max_images > 1 else 10000
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp
        self.topil = torchvision.transforms.ToPILImage()
        self.selected_keys = selected_keys
        self.image_size = image_size



    def log_wandb_image(self, data, logger, split, nrow, caption, step):
        
        if len(list(data.keys())) == 0:
            return 

        image_row = list() 
        keys = list(data.keys()) + ['caption']
        for k, images in data.items():
            image_row.append(images)

        
        grid = list() 
        for row in zip(*image_row):
            grid.append(torch.stack(row))
        key = " | ".join(list(data.keys()))
        logger.log_image(key=f"{split}/{key}", images=grid, caption=caption)


    def log_wandb_model(self, model, logger):
        logger.watch(model,)
    


    @rank_zero_only
    def log_experiment(self, pl_model, data, split, nrow, caption):
        # image/video: b, t, c, w, h
        for logger in pl_model.loggers:
                       
            if isinstance(logger, pytorch_lightning.loggers.wandb.WandbLogger):
                self.log_wandb_image(data, logger, split, nrow, caption, pl_model.global_step)



    def log_img(self, pl_module, batch, batch_ids, split="train"):
        if (self.check_frequency(batch_ids) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                batch_data, b, caption = pl_module.log_images(batch)
            
            selected_batch_idx = random.sample(range(b), min(self.max_images, b))
            selected_batch = dict()
            selected_keys = self.selected_keys if self.selected_keys is not None else list(batch_data.keys())

            for k, data in batch_data.items():
                if k in selected_keys:
                    resize_data = torch.nn.functional.interpolate(data, (512, 512)).detach().cpu()
                    selected_batch[k] = resize_data

            
            self.log_experiment(pl_module, selected_batch, split, b, caption)

            if is_train:
                pl_module.train()

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs):
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, *args, **kwargs):
        self.log_img(pl_module, batch, batch_idx, split="val") 

            
