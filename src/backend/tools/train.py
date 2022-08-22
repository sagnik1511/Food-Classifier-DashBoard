import json
from os import path as osp
from ..datasets import FoodDataset
from torch_tutor.core.trainer import Trainer
from torch_tutor.core.callbacks import CallBack
from ..models import *
from torch.optim import *
import torch.nn as nn
import argparse

parser = argparse.ArgumentParser(description="Training Configurations")

parser.add_argument('-c', "--config", type=str, default="1x_gpu_base_scheduler.json",
    metavar="", help="training config")
parser.add_argument("-d", "--dataset", type=str, default="food_base.json",
    metavar="", help="dataset config")
parser.add_argument("-m", "--model", default="foodnet_base.json",
    metavar="", help="model config")



def trigger_scheduler(scheduler_config, dataset_config, model_config):
    print(f"Job Initiated: {scheduler_config['job_name']}")

    print("Loading dataset object")
    train_ds = FoodDataset(dataset_config["path"]["root_dir"],
                            dataset_config["path"]["train"],
                            **dataset_config["data_shape"])
    val_ds = FoodDataset(dataset_config["path"]["root_dir"],
                            dataset_config["path"]["validation"],
                            **dataset_config["data_shape"])

    if model_config["name"] == "foodnet_base":
        model = FoodNet(model_config, dataset_config["num_classes"])
    callback = CallBack(**scheduler_config["callback"])
    trainer = Trainer(train_ds, model, scheduler_config["device"])
    optim_name = scheduler_config["optimizer"]["name"]
    if optim_name == "adam":
        optimizer = Adam
    else:
        raise NotImplementedError
    
    if scheduler_config["loss"] == "cross_entropy":
        loss_fn = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError

    trainer.compile(optimizer, loss_fn,
                    scheduler_config["metrics"],
                    scheduler_config["optimizer"]["params"])

    trainer.train(scheduler_config["batch_size"],
                    scheduler_config["max_epochs"],
                    validation_set=val_ds, callback=callback)



if __name__ == '__main__':
    args = parser.parse_args()
    scheduler_config = json.load(open(osp.join("src/backend/configs/training", args.config), "r"))
    dataset_config = json.load(open(osp.join("src/backend/configs/datasets", args.dataset), "r"))
    model_config = json.load(open(osp.join("src/backend/configs/models", args.model), "r"))
    trigger_scheduler(scheduler_config, dataset_config, model_config)
    