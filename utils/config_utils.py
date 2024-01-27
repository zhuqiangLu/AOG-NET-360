import importlib
import argparse
from omegaconf import OmegaConf
from datetime import datetime
import os 



def get_obj_from_str(string:str, reload:bool=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)



def instantiate_from_config(config:dict):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")

    return get_obj_from_str(config["target"])(**config.get("params", dict()))



def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        help="project name",
        default="my_project",
    )

    parser.add_argument(
        "-wandb_id",
        type=str,
        default="",
        help="wandb project description, used as name"
    )

   
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=114514,
        help="seed for seed_everything",
    )

    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        default=None,
        help="checkpoint for thr model"
    )

    parser.add_argument(
        "--num_inference_step",
        type=int,
        default=25,
        help="number of inference step"
    )

    parser.add_argument(
        "--inference_out_dir",
        type=str,
        default='inference_result',
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=2.5,
        help="cfg_scale"
    )
    parser.add_argument(
        "-w",
        "--wandb",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enbale wandb logger with tensorboard",

    )

    


    
    return parser



def get_configs(build_dir=True, **kwargs):

    default_parser = get_parser(**kwargs)
    opt, unknown = default_parser.parse_known_args()

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    # correct learning rate 
    # base_lr = config.model.params.lr 
    # bs = config.data.params.batch_size
    # accumulate = config.trainer.accumulate_grad_batches
    # ngpu = config.trainer.gpus

    # config.model.params.lr = accumulate * (ngpu/8.) * (bs/4.) * base_lr

    # print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus/8) * {} (batchsize/4) * {:.2e} (base_lr)".format(
    #     config.model.params.lr, accumulate, ngpu/8, bs/4, base_lr))
    
    # check resume 
    config.trainer.default_root_dir = os.path.join(config.trainer.default_root_dir, opt.name)
    return opt, config, 

