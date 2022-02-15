import torch
import pytorch_lightning as pl

import os, sys

# third-party library
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import dataset_processing
from timeit import default_timer as timer
from utils.report import report_precision_se_sp_yi, report_mae_mse
from utils.utils import Logger, AverageMeter, time_to_str, weights_init
from utils.genLD import genLD
from model.resnet50 import resnet50
import torch.backends.cudnn as cudnn
from transforms.affine_transforms import *
import time
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class DataRefactoring(pl.LightningDataModule):

  def __init__(self, BATCH_SIZE, BATCH_SIZE_TEST, NUM_WORKERS, DATA_PATH, TRAIN_FILE, TEST_FILE):
    super().__init__()
    self.batch_size=BATCH_SIZE
    self.batch_size_test = BATCH_SIZE_TEST
    self.num_workers = NUM_WORKERS
    self.data_path = DATA_PATH
    self.train_file = TRAIN_FILE
    self.test_file = TEST_FILE
    self.normalize = transforms.Normalize(mean=[0.45815152, 0.361242, 0.29348266],
                                     std=[0.2814769, 0.226306, 0.20132513])

  def setup(self, stage):
    self.dset_train = dataset_processing.DatasetProcessing(
        self.data_path, self.train_file, transform=transforms.Compose([
                transforms.Scale((256, 256)),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                RandomRotate(rotation_range=20),
                self.normalize,
            ]))
    self.dset_test = dataset_processing.DatasetProcessing(
        self.data_path, self.test_file, transform=transforms.Compose([
                transforms.Scale((224, 224)),
                transforms.ToTensor(),
                self.normalize,
            ]))

  def train_dataloader(self):
    train_loader = DataLoader(self.dset_train,
                              batch_size=self.batch_size,
                              shuffle=True,
                              num_workers=self.num_workers,
                              pin_memory=True)
    return train_loader

  def test_dataloader(self):
    test_loader = DataLoader(self.dset_test,
                             batch_size=self.batch_size_test,
                             shuffle=False,
                             num_workers=self.num_workers,
                             pin_memory=True)
    return test_loader

class ModelRefactoring(pl.LightningModule):
  def __init__(self, LR, sigma, lmbd):
    super().__init__()
    self.save_hyperparameters()
    self.cnn = resnet50()
    cudnn.benchmark = True

  def forward(self, x):
    cls, cou, cou2cls = self.cnn(x, None)
    return cls, cou, cou2cls

  def loss(self, true_cls, true_cou, true_cou2cls, pred_ld4, pred_ld):
    #change to torch scheduler
    loss_func = nn.CrossEntropyLoss()
    kl_loss_1 = nn.KLDivLoss()
    kl_loss_2 = nn.KLDivLoss()
    kl_loss_3 = nn.KLDivLoss()

    # print('Predicted by CNN: 1', true_cls, '2 ', true_cou, '3 ', true_cou2cls)
    # print('Label distributions 1: ', pred_ld4,'2 ', pred_ld)
    # print(torch.log(true_cls), pred_ld4)
    loss_cls = kl_loss_1(torch.log(true_cls), pred_ld4) * 4.0
    loss_cou = kl_loss_2(torch.log(true_cou), pred_ld) * 65.0
    loss_cls_cou = kl_loss_3(torch.log(true_cou2cls), pred_ld4) * 4.0
    loss = (loss_cls + loss_cls_cou) * 0.5 * self.hparams.lmbd + loss_cou * (1.0 - self.hparams.lmbd)
    return loss

  def configure_optimizers(self):    
    # print(self.hparams)
    # print(var(self.hparams))
    # print(self.hparams.LR)
    optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams['LR'], momentum=0.9, weight_decay=5e-4)    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    return {'optimizer' : optimizer, 
            'lr_scheduler' : scheduler}

  def training_step(self, batch, batcn_num):
    # train
    x, y, l = batch
    # x = x
    # l = l.cpu().numpy()

    # generating ld
    l = l - 1
    ld = genLD(l, self.hparams.sigma, 'klloss', 65)
    ld_4 = np.vstack((np.sum(ld[:, :5], 1), np.sum(ld[:, 5:20], 1), np.sum(ld[:, 20:50], 1), np.sum(ld[:, 50:], 1))).transpose()
    ld = torch.from_numpy(ld).float().to(device)
    ld_4 = torch.from_numpy(ld_4).float().to(device)

    cls, cou, cou2cls = self.forward(x) #nn output
    train_loss = self.loss(cls, cou, cou2cls, ld_4, ld)
    # print(message)
    self.log('train_loss', train_loss, on_epoch=True, prog_bar=True, logger=True)
    return train_loss

  def validation_step(self, val_batch, batch_idx):
    x, y, l = val_batch
    
    x = x 
    y = y

    # y_true = y.data.cpu().numpy()
    # l_true = l.data.cpu().numpy()

    cls, cou, cou2cls = self.forward(x)

    val_loss = self.loss_func(cou2cls, y)

    _, preds_m = torch.max(cls + cou2cls, 1)
    _, preds = torch.max(cls, 1)
    # y_pred = preds.data.cpu().numpy()
    # y_pred_m = preds_m.data.cpu().numpy()

    _, preds_l = torch.max(cou, 1)
    preds_l = (preds_l + 1)#.data.cpu().numpy()
    # preds_l = cou2cou.data.cpu().numpy()
    l_pred = preds_l

    batch_corrects = torch.sum((preds == y))
    
    self.log("performance", {"val_loss": val_loss, "test_corrects": batch_corrects}, prog_bar=True, logger=True)
    return {'y_true': y_true, 
            'l_true': l_true, 
            'y_pred': y_pred, 
            'y_pred_m': y_pred_m, 
            'l_pred': l_pred,
            }

  def validation_epoch_end(self, validation_step_outputs):
    
    l_true = np.hstack(validation_step_outputs['l_true'])
    l_pred = np.hstack(validation_step_outputs['l_pred'])

    y_true = np.hstack(validation_step_outputs['y_true'])
    y_pred = np.hstack(validation_step_outputs['y_pred'])
    y_pred_m = np.hstack(validation_step_outputs['y_pred_m'])

    _, _, pre_se_sp_yi_report = report_precision_se_sp_yi(y_pred, y_true)
    _, _, pre_se_sp_yi_report_m = report_precision_se_sp_yi(y_pred_m, y_true)
    _, MAE, MSE, mae_mse_report = report_mae_mse(l_true, l_pred, y_true)

    self.log('metrics', {'pre_se_sp_yi_report' : pre_se_sp_yi_report,
                        'pre_se_sp_yi_report_m': pre_se_sp_yi_report_m,
                        'MAE': MAE, 
                        'MSE': MSE,
                        'mae_mse_report': mae_mse_report},  prog_bar=True, logger=True)





