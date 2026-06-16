
from argparse import ArgumentParser

import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies.ddp import DDPStrategy

from model.utils import *


def parseArgs():
    parser = ArgumentParser()
    #TODO diffusion standard config
    parser.add_argument('--config_model', type=str,default="CaFFeGru.yaml") #TODO insert the model config
    parser.add_argument('--config_data', type=str, default="temporalTest.yaml") #TODO insert the data config
    parser.add_argument('--reset_optimizers_auto', action='store_true',default=False)
    parser.add_argument('--name', type=str, default="--default")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--accumulated_grad_batches', type=int, default=128)

    parser.add_argument('--devices', type=int, default=1)
    parser.add_argument('--parent_dir_for_validation', type=str, default= None) #TODO insert your path
    parser.add_argument('--logdir', type=str, default='log_dir')


    return parser.parse_args()



if __name__ == "__main__":

    cfg = parseArgs()

    batch_size = cfg.batch_size # x8 for temporal x 16 for runtime patching
    devices = cfg.devices
    accumulated_grad_batches = cfg.accumulated_grad_batches

    parent_dir_for_validation = cfg.parent_dir_for_validation

    config_model = cfg.config_model
    config_data = cfg.config_data

    batch_size = int(batch_size / (devices * accumulated_grad_batches))

    log_dir = cfg.logdir

    model = instantiate_completely("model", config_model, parent_dir_for_validation=parent_dir_for_validation)
    gdm = instantiate_completely("Data", config_data, batch_size=batch_size)

    logger = TensorBoardLogger(save_dir=log_dir, name=config_model[:-5] + "_" + config_data[:-5])
    print("********************************************************************************")
    print("Dataloader file: ", config_data)
    print("Model file: ", config_model)
    print("********************************************************************************")

    mc = ModelCheckpoint(save_top_k=1, monitor="val/mde", mode="max",
                         filename='{epoch}-{' + "val/mde" + ':.4f}')
    mc2 = ModelCheckpoint()

    cb = [mc, mc2]


    args = OmegaConf.load(get_yaml("model", config_model))
    config_dict_model = get_dict_from_model_config(args)
    logger.log_hyperparams({**config_dict_model})
    trainer = pl.Trainer(max_epochs=80, accelerator="gpu", devices=devices, logger=logger,
                                     accumulate_grad_batches=accumulated_grad_batches, callbacks=cb)

    args = OmegaConf.load(get_yaml("model", config_model))

    if args.get("ckpt") is not None:
        ckpt_optimizers = args["ckpt"]
    else:
        ckpt_optimizers = None

    trainer.fit(model, train_dataloaders=gdm.train_dataloader(), val_dataloaders=gdm.val_dataloader(),
                ckpt_path=ckpt_optimizers)


