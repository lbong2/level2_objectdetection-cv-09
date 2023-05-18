import os

import torch
#from tensorboardX import SummaryWriter
import wandb

from config import train_config
from dataloader.dataset import CustomDataset, split_train_valid, get_all_annotation, get_json_data
from utils.evaluate_utils import evaluate
from utils.plot_utils import plot_loss_and_lr, plot_map
from utils.train_utils import train_one_epoch, create_model
from optimizers.optims import create_optimizer
from optimizers.schedulers import create_scheduler

import albumentations as A
from albumentations.pytorch import ToTensorV2

import matplotlib.pyplot as plt
import argparse
import numpy as np
import random

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def main():
    cfg = train_config.default_config
    if cfg["wandb"]:
        wandb.init(
            entity=cfg['entity'],
            project=cfg['project'],
            name=cfg['name'],
            config=cfg,
        )
        cfg = wandb.config
    cfg = train_config.ChangeConfig(cfg)
    # wandb config
    cfg.model_save_dir = os.path.join(cfg.model_save_dir,cfg.project,cfg.name)
    seed_everything(cfg.seed)

    device = torch.device(cfg.device)
    print("Using {} device training.".format(device.type))
  
    if not os.path.exists(cfg.model_save_dir):
        os.makedirs(cfg.model_save_dir)

    data_transform = {
        "train": A.Compose([
            A.HorizontalFlip(p=cfg.train_horizon_flip_prob),
            A.RandomBrightnessContrast(p=0.2,brightness_limit=0.5, contrast_limit=0.5),
            A.ShiftScaleRotate(scale_limit=[-0.3,0.3]),
            # A.RandomCrop(1024,1024),
            # A.Resize(1024,1024),
            # A.OneOf([
            #     A.CLAHE(p=0.3),
            #     A.Blur(blur_limit=(30, 40), p=0.5)
            # ], p=0.3),
            ToTensorV2(),
        ], bbox_params=A.BboxParams(format='pascal_voc',label_fields=['labels'])),
        "val": A.Compose([
            ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc',label_fields=['labels']))
    }

    if not os.path.exists(cfg.data_root_dir):
        raise FileNotFoundError("dataset root dir not exist!")
    json_data = get_json_data(cfg.data_root_dir)
    
    json_anno = get_all_annotation(json_data)

    train_info, val_info = split_train_valid(cfg.data_root_dir, json_data)

    # load train data set
    train_data_set = CustomDataset(train_info, json_anno, data_transform['train'], mosaic=cfg.mosaic)
    batch_size = cfg.batch_size
    if cfg.num_workers == False:
        cfg.num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers'.format(cfg.num_workers))
    train_data_loader = torch.utils.data.DataLoader(train_data_set,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=cfg.num_workers,
                                                    collate_fn=train_data_set.collate_fn)

    # load validation data set
    val_data_set = CustomDataset(val_info, json_anno, data_transform['val'])
    val_data_set_loader = torch.utils.data.DataLoader(val_data_set,
                                                      batch_size=batch_size,
                                                      shuffle=False,
                                                      num_workers=cfg.num_workers,
                                                      collate_fn=train_data_set.collate_fn)

    # create model num_classes equal background + 80 classes
    model = create_model(num_classes=cfg.num_class,cfg=cfg)

    model.to(device)

    # define optimizer
    optimizer = create_optimizer(
            cfg.optimizer, 
            filter(lambda p: p.requires_grad, model.parameters()), 
            cfg
            )
        
    lr_scheduler = create_scheduler(cfg.scheduler,optimizer,cfg)

    # train from pretrained weights
    if cfg.resume != "":
        checkpoint = torch.load(cfg.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        cfg.start_epoch = checkpoint['epoch'] + 1
        print("the training process from epoch{}...".format(cfg.start_epoch))

    train_loss = []
    learning_rate = []
    train_mAP_list = []
    val_mAP = []
    early_stop_count = 0
    best_mAP = 0
    for epoch in range(cfg.start_epoch, cfg.num_epochs):
        loss_dict, total_loss = train_one_epoch(model, optimizer, train_data_loader,
                                                device, epoch, train_loss=train_loss, train_lr=learning_rate,
                                                print_freq=50, warmup=False, loss_gain=cfg.loss_gain)


        print("------>Starting training data valid")
        train_aps, train_mAP = evaluate(model, train_data_loader, device=device, mAP_list=train_mAP_list)

        print("------>Starting validation data valid")
        val_aps, mAP = evaluate(model, val_data_set_loader, device=device, mAP_list=val_mAP)
        print('training mAp is {}'.format(train_mAP))
        print('validation mAp is {}'.format(mAP))
        print('best mAp is {}'.format(best_mAP))

        if cfg.scheduler == 'reducelronplateau':
            lr_scheduler.step(mAP)
        else:
            lr_scheduler.step()


        # wandb log - by kyungbong
        if cfg.wandb:
            # plot metric 및 loss
            metric_board_info = {
                'metric/train_mAP': train_mAP,
                'metric/val_mAP': mAP,
            }

            train_board_info = {'train/lr': optimizer.param_groups[0]['lr']}
            for k, v in loss_dict.items():
                train_board_info["train/"+k] = v.item()
            train_board_info['train/total loss'] = total_loss.item()
            train_board_info['train/epoch'] = epoch

            wandb.log(train_board_info, step= epoch)
            wandb.log(metric_board_info, step=epoch)

            train_table = wandb.Table(columns=["ap","class"],
                                data =[[x,y] for x,y in zip(train_aps,CustomDataset.classes)])
            val_table = wandb.Table(columns=["ap","class"],
                                data = [[x,y] for x,y in zip(val_aps,CustomDataset.classes)])
            wandb.log({
                'hist/trian_APs': wandb.plot.histogram(train_table, value='ap', title='APs by Class'),
                'hist/val_APs': wandb.plot.histogram(val_table, value='ap', title='APs by Class')
                }, step = epoch)
            # wandb.log({'val_ap for class': wandb.plot.histogram(val_table)})


        if mAP > best_mAP:
            early_stop_count = 0
            best_mAP = mAP
            # save weights
            save_files = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch}
            if not os.path.exists(cfg.model_save_dir):
                os.makedirs(cfg.model_save_dir)
            
            torch.save(save_files,
                       os.path.join(cfg.model_save_dir, "best.pth"))
            print(f"save {cfg.model_save_dir}/best.pth")
        else:
            early_stop_count +=1
            if early_stop_count > cfg.early_stop:
                print("Early Stopping")
                break
    # plot loss and lr curve
    
    if len(train_loss) != 0 and len(learning_rate) != 0:
        plot_loss_and_lr(train_loss, learning_rate, cfg.model_save_dir)
    
    # plot mAP curve
    if len(val_mAP) != 0:
        plot_map(val_mAP, cfg.model_save_dir)
        
    # wandb plot image log - by kyungbong
    if cfg.wandb:
        wandb.log({
            "mAPCurve": wandb.Image(plt.imread(os.path.join(cfg.model_save_dir, "mAP.png"))),
            "loss_lrCurve": wandb.Image(plt.imread(os.path.join(cfg.model_save_dir, "loss_and_lr.png"))),
        })
        wandb.finish()     

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep",action='store_true',help="using wandb sweep")
    args = parser.parse_args()

    version = torch.version.__version__[:5]
    print('torch version is {}'.format(version))

    if args.sweep:
        sweep_id = wandb.sweep(
            sweep = train_config.sweep_config,
        )
        wandb.agent(sweep_id, function=main)
    else:
        main()
