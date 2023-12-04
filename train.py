import os
import time
import datetime
import yaml
import torch
import torch.nn as nn
import torch.utils.data as data
import argparse

from dataset import CityscapesDataset
from utils import MixSoftmaxCrossEntropyOHEMLoss, MixSoftmaxCrossEntropyLoss, EncNetLoss, IterationPolyLR, WarmupPolyLR, SegmentationMetric_MY, SetupLogger
from models import CSFCN
import torch.backends.cudnn as cudnn

cudnn.enabled = True

cudnn.benchmark = True

cudnn.deterministic = True


import numpy as np
import random
#
# fix all random seeds
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
np.random.seed(123)
random.seed(123)
# torch.backends.cudnn.deterministic = True
# #  torch.backends.cudnn.benchmark = True
# #  torch.multiprocessing.set_sharing_strategy('file_system')


def set_optimizer(model):
    if hasattr(model, 'get_params'):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        params_list = [
            {'params': wd_params, },
            {'params': nowd_params, 'weight_decay': 0},
            {'params': lr_mul_wd_params, 'lr': 1e-2 * 10},
            {'params': lr_mul_nowd_params, 'weight_decay': 0, 'lr': 1e-2 * 10},
        ]
    else:
        wd_params, non_wd_params = [], []
        for name, param in model.named_parameters():
            if param.dim() == 1:
                non_wd_params.append(param)
            elif param.dim() == 2 or param.dim() == 4:
                wd_params.append(param)
        params_list = [
            {'params': wd_params, },
            {'params': non_wd_params, 'weight_decay': 0},
        ]

    optim = torch.optim.SGD(
        params_list,
        lr=1e-2,
        momentum=0.9,
        weight_decay=5e-4,
    )
    return optim



class Trainer(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataparallel = torch.cuda.device_count() > 1
        
        # dataset and dataloader
        train_dataset = CityscapesDataset(root = cfg["train"]["cityscapes_root"], 
                                          split='train', 
                                          base_size=cfg["model"]["base_size"], 
                                          crop_size=cfg["model"]["crop_size"])
        val_dataset = CityscapesDataset(root = cfg["train"]["cityscapes_root"], 
                                        split='val',
                                        base_size=cfg["model"]["base_size"], 
                                        crop_size=cfg["model"]["crop_size"])
        self.train_dataloader = data.DataLoader(dataset=train_dataset,
                                                batch_size=cfg["train"]["train_batch_size"],
                                                shuffle=True,
                                                num_workers=4,
                                                pin_memory=True,
                                                drop_last=False)
        self.val_dataloader = data.DataLoader(dataset=val_dataset,
                                              batch_size=cfg["train"]["valid_batch_size"],
                                              shuffle=False,
                                              num_workers=4,
                                              pin_memory=True,
                                              drop_last=False)
        
        self.iters_per_epoch = len(self.train_dataloader)
        self.max_iters = cfg["train"]["epochs"] * self.iters_per_epoch

        # create network
        self.model = CSFCN(n_classes = train_dataset.NUM_CLASS).to(self.device)

        pretrained_dict = torch.load(cfg["train"]["pretrained"])

        model_dict = self.model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        print("load model pth")

        self.criterion = MixSoftmaxCrossEntropyOHEMLoss(ignore_index=train_dataset.IGNORE_INDEX).to(self.device)

        self.optimizer = set_optimizer(self.model)

        self.lr_scheduler = WarmupPolyLR(self.optimizer,
                                            max_iters=self.max_iters,
                                            power=0.9)

        if(self.dataparallel):
             self.model = nn.DataParallel(self.model)

        # evaluation metrics
        self.metric = SegmentationMetric_MY(train_dataset.NUM_CLASS)

        self.current_mIoU = 0.0
        self.best_mIoU = 0.0
        self.time = 100

        self.epochs = cfg["train"]["epochs"]
        self.current_epoch = 0
        self.current_iteration = 0
        
    def train(self):
        epochs, max_iters = self.epochs, self.max_iters
        log_per_iters = self.cfg["train"]["log_iter"]
        val_per_iters = self.cfg["train"]["val_epoch"] * self.iters_per_epoch
        
        start_time = time.time()
        logger.info('Start training, Total Epochs: {:d} = Total Iterations {:d}'.format(epochs, max_iters))
        
        self.model.train()
        
        for _ in range(self.epochs): 
            self.current_epoch += 1
            lsit_pixAcc = []
            list_mIoU = []
            list_loss = []
            self.metric.reset()
            for i, (images, targets, _) in enumerate(self.train_dataloader):  
                self.current_iteration += 1
                
                self.lr_scheduler.step()

                images = images.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
		
                self.metric.update(outputs[0], targets)
                pixAcc, mIoU = self.metric.get()
                lsit_pixAcc.append(pixAcc)
                list_mIoU.append(mIoU)
                list_loss.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                eta_seconds = ((time.time() - start_time) / self.current_iteration) * (max_iters - self.current_iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                
                if self.current_iteration % log_per_iters == 0:
                    logger.info(
                        "Epochs: {:d}/{:d} || Iters: {:d}/{:d} || Lr: {:.6f} || Loss: {:.4f} || mIoU: {:.4f} || Cost Time: {} || Estimated Time: {}".format(
                            self.current_epoch, self.epochs, 
                            self.current_iteration, max_iters, 
                            self.optimizer.param_groups[0]['lr'], 
                            loss.item(), 
                            mIoU,
                            str(datetime.timedelta(seconds=int(time.time() - start_time))), 
                            eta_string))
		
            average_pixAcc = sum(lsit_pixAcc)/len(lsit_pixAcc)
            average_mIoU = sum(list_mIoU)/len(list_mIoU)
            average_loss = sum(list_loss)/len(list_loss)
            logger.info("Epochs: {:d}/{:d}, Average loss: {:.4f}, Average mIoU: {:.4f}, Average pixAcc: {:.4f}".format(self.current_epoch, self.epochs, average_loss, average_mIoU, average_pixAcc))
		
            if self.current_iteration % val_per_iters == 0:
                self.validation()
                self.model.train()

        total_training_time = time.time() - start_time
        total_training_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f}s / it)".format(
            total_training_str, total_training_time / max_iters))

    def validation(self):
        is_best = False
        self.metric.reset()
        if self.dataparallel:
            model = self.model.module
        else:
            model = self.model
        model.eval()
        lsit_pixAcc = []
        list_mIoU = []
        # list_loss = []
        total_time = []
        for i, (image, targets, filename) in enumerate(self.val_dataloader):
            image = image.to(self.device)
            targets = targets.to(self.device)

            with torch.no_grad():
                start = time.time()
                outputs = model(image)
                total_time.append(time.time() - start)
                # loss = self.criterion(outputs, targets)
            self.metric.update(outputs[0], targets)
            pixAcc, mIoU = self.metric.get()
            lsit_pixAcc.append(pixAcc)
            list_mIoU.append(mIoU)
            # list_loss.append(loss.item())
        mean_time = sum(total_time) / len(total_time)
        average_pixAcc = sum(lsit_pixAcc) / len(lsit_pixAcc)
        average_mIoU = sum(list_mIoU) / len(list_mIoU)
        # average_loss = sum(list_loss)/len(list_loss)
        self.current_mIoU = average_mIoU
        if self.time >= mean_time:
            self.time = mean_time
        logger.info(
            "Validation: Average mIoU: {:.4f}, Average pixAcc: {:.4f}, Best mIoU: {:.4f}, Best Inference Time: {:.4f}".format(
                average_mIoU, average_pixAcc, self.best_mIoU, self.time))

        if self.current_mIoU > self.best_mIoU and self.current_mIoU > 0.75:
            is_best = True
            self.best_mIoU = self.current_mIoU
        if is_best:
            save_checkpoint(self.model, self.cfg, self.current_epoch, is_best, self.current_mIoU, self.dataparallel)
        
def save_checkpoint(model, cfg, epoch = 0, is_best=False, mIoU = 0.0, dataparallel = False):
    """Save Checkpoint"""
    directory = os.path.expanduser(cfg["train"]["ckpt_dir"])
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = '{}_{}_{}_{:.5f}.pth'.format(cfg["model"]["name"], cfg["model"]["backbone"],epoch,mIoU)
    filename = os.path.join(directory, filename)
    if dataparallel:
        model = model.module
    if is_best:
        best_filename = '{}_{}_{}_{:.5f}_best_model.pth'.format(cfg["model"]["name"], cfg["model"]["backbone"],epoch,mIoU)
        best_filename = os.path.join(directory, best_filename)
        torch.save(model.state_dict(), best_filename)
        

if __name__ == '__main__':
    # Set config file
    parser = argparse.ArgumentParser(description="Pytorch Real-time Semantic Segmentation Training")
    parser.add_argument("-cfg",
                        "--config-file",
                        default="",
                        metavar="FILE",
                        help="path to config file",
                        type=str)
    args = parser.parse_args()

    config_path = args.config_file
    with open(config_path, "r") as yaml_file:
        cfg = yaml.load(yaml_file.read())

    # Use specific GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["train"]["specific_gpu_num"])
    num_gpus = len(cfg["train"]["specific_gpu_num"].split(','))
    print("torch.cuda.is_available(): {}".format(torch.cuda.is_available()))
    print("torch.cuda.device_count(): {}".format(torch.cuda.device_count()))
    print("torch.cuda.current_device(): {}".format(torch.cuda.current_device()))

    # Set logger
    logger = SetupLogger(name = "semantic_segmentation", 
                          save_dir = cfg["train"]["ckpt_dir"], 
                          distributed_rank = 0,
                         filename='{}_{}_{}_{}_log.txt'.format(
                             cfg["model"]["name"], cfg["model"]["backbone"], 'cityscapes',
                             time.strftime(r"%Y_%m_%d_%H_%M_%S")))


    logger.info("Using {} GPUs".format(num_gpus))
    logger.info("torch.cuda.is_available(): {}".format(torch.cuda.is_available()))
    logger.info("torch.cuda.device_count(): {}".format(torch.cuda.device_count()))
    logger.info("torch.cuda.current_device(): {}".format(torch.cuda.current_device()))
    logger.info(cfg)


    # Start train
    trainer = Trainer(cfg)
    trainer.train()
    
