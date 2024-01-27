from utils.config_utils import get_configs, instantiate_from_config
from omegaconf import OmegaConf
from pprint import pprint
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import warnings
import os
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
warnings.filterwarnings("ignore", message="The pts_unit 'pts' gives wrong results. Please use pts_unit 'sec'.")
import torch
torch.backends.cuda.matmul.allow_tf32 = True

if os.name == 'nt':
    ddp = DDPStrategy(process_group_backend="gloo")

def train():
    parser, config = get_configs() 
    # seed 
    pl.seed_everything(parser.seed)
    # build model
    model_config = config.get('model', OmegaConf.create())
    model = instantiate_from_config(model_config)
    if parser.checkpoint is not None:
        model.init_from_ckpt(parser.checkpoint)

    # build data 
    data_config = config.get('data', OmegaConf.create())
    data = instantiate_from_config(data_config)

    callbacks = []
    callbacks_configs = config.get('callbacks', OmegaConf.create())
    for _, v in callbacks_configs.items():
        callbacks.append(instantiate_from_config(v))
    
    # build trainer 
    trainer_config = config.get("trainer", OmegaConf.create())
    trainer_config = argparse.Namespace(**trainer_config)
    
    # init logger
    loggers = [TensorBoardLogger(trainer_config.default_root_dir)]
    log_dir =  loggers[-1].log_dir
    if parser.wandb:
        loggers.append(WandbLogger(save_dir=trainer_config.default_root_dir, project=parser.name, name=parser.wandb_id))


    trainer_sub_args = {
        'callbacks': callbacks,
        'logger': loggers,
        # 'plugins': list()
    }

    # set window backend as gloo
    backend_name = 'ddp'
    if os.name == 'nt':
        backend_name = 'gloo'
        ddp = DDPStrategy(process_group_backend=backend_name, find_unused_parameters=True)
    else:
        ddp = DDPStrategy(find_unused_parameters=True)

    trainer_sub_args['strategy'] = ddp
    trainer = Trainer.from_argparse_args(trainer_config, **trainer_sub_args)

    for logger in loggers:
        if isinstance(logger, WandbLogger):
            logger.watch(model)
            # logger.experiment.config["model_config"] = config
            # logger.experiment.config["local_checkpoint_dir"] = log_dir

    trainer.fit(model, data, )


  



if __name__ == "__main__":


    train()
