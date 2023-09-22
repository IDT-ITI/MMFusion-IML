"""
Created by Kostas Triaridis (@kostino)
in August 2023 @ ITI-CERTH
"""
import os
import argparse
import numpy as np
from tqdm import tqdm
from common.utils import AverageMeter
from common.losses import TruForLossPhase2
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import logging
import torch
import torchvision.transforms.functional as TF


from data.datasets import MixDataset
from common.metrics import computeDetectionMetrics
from models.ws_cmnext_conf import WSCMNeXtWithConf
from common.split_params import group_weight
from common.lr_schedule import WarmUpPolyLR
from models.modal_extract import ModalitiesExtractor
from configs.cmnext_init_cfg import _C as config, update_config

parser = argparse.ArgumentParser(description='')
parser.add_argument('-gpu', '--gpu', type=int, default=0, help='device, use -1 for cpu')
parser.add_argument('-log', '--log', type=str, default='INFO', help='logging level')
parser.add_argument('-exp', '--exp', type=str, default=None, help='Yaml experiment file')
parser.add_argument('-ckpt', '--ckpt', type=str, default=None, help='Localization checkpoint')
parser.add_argument('opts', help="other options", default=None, nargs=argparse.REMAINDER)

args = parser.parse_args()

config = update_config(config, args.exp)

gpu = args.gpu
loglvl = getattr(logging, args.log.upper())
logging.basicConfig(level=loglvl, format='%(message)s')

device = 'cuda:%d' % gpu if gpu >= 0 else 'cpu'
np.set_printoptions(formatter={'float': '{: 7.3f}'.format})

if device != 'cpu':
    # cudnn setting
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED


modal_extractor = ModalitiesExtractor(config.MODEL.MODALS[1:], config.MODEL.NP_WEIGHTS)

model = WSCMNeXtWithConf(config.MODEL)

ckpt = torch.load(args.ckpt)

model.load_state_dict(ckpt['state_dict'], strict=False)
modal_extractor.load_state_dict(ckpt['extractor_state_dict'])


modal_extractor.to(device)
model = model.to(device)


train = MixDataset(config.DATASET.TRAIN,
                   config.DATASET.IMG_SIZE,
                   train=True,
                   class_weight=config.DATASET.CLASS_WEIGHTS)

val = MixDataset(config.DATASET.VAL,
                 config.DATASET.IMG_SIZE,
                 train=False)

logging.info(train.get_info())
train_loader = DataLoader(train,
                          batch_size=config.BATCH_SIZE,
                          shuffle=True,
                          num_workers=config.WORKERS,
                          pin_memory=True)

val_loader = DataLoader(val,
                        batch_size=1,
                        shuffle=False,
                        num_workers=config.WORKERS,
                        pin_memory=True)

criterion = TruForLossPhase2()

os.makedirs('./ckpt/{}'.format(config.MODEL.NAME), exist_ok=True)
logdir = './{}/{}'.format(config.LOG_DIR, config.MODEL.NAME)
os.makedirs(logdir, exist_ok=True)
writer = SummaryWriter('./{}/{}'.format(config.LOG_DIR, config.MODEL.NAME))

cmnext_params = []
cmnext_params = group_weight(cmnext_params, model, torch.nn.BatchNorm2d, config.LEARNING_RATE)
params = cmnext_params

optimizer = torch.optim.SGD(params,
                            lr=config.LEARNING_RATE,
                            momentum=config.SGD_MOMENTUM,
                            weight_decay=config.WD
                            )

iters_per_epoch = len(train_loader)
iters = 0
max_iters = config.EPOCHS * iters_per_epoch
min_loss = 100

lr_schedule = WarmUpPolyLR(optimizer,
                           start_lr=config.LEARNING_RATE,
                           lr_power=0.9,
                           total_iters=max_iters,
                           warmup_steps=iters_per_epoch * config.WARMUP_EPOCHS)

scaler = torch.cuda.amp.GradScaler()

for epoch in range(config.EPOCHS):
    train.shuffle()  # for balanced sampling
    model.set_train()
    modal_extractor.set_val()
    avg_loss = AverageMeter()
    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(train_loader, desc='Training Epoch {}/{}'.format(epoch + 1, config.EPOCHS), unit='steps')
    for step, (images, _, masks, labels) in enumerate(pbar):

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        masks = masks.squeeze(1).to(device, non_blocking=True)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            modals = modal_extractor(images)

            images_norm = TF.normalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            inp = [images_norm] + modals

            anomaly, confidence, detection = model(inp)

            loss = criterion(anomaly, masks, confidence, detection, labels) / config.ACCUMULATE_ITERS

        scaler.scale(loss).backward()
        if ((step + 1) % config.ACCUMULATE_ITERS == 0) or (step + 1 == len(train_loader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        avg_loss.update(loss.detach().item())

        curr_iters = epoch * iters_per_epoch + step

        lr_schedule.step(cur_iter=curr_iters)
        writer.add_scalar('Total Loss', loss.detach().item(), curr_iters)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], curr_iters)

        pbar.set_postfix({"last_loss": loss.detach().item(), "epoch_loss": avg_loss.average()})

    scores = []
    labels = []
    val_loss_avg = AverageMeter()
    model.set_val()
    modal_extractor.set_val()
    pbar = tqdm(val_loader, desc='Validating Epoch {}/{}'.format(epoch + 1, config.EPOCHS), unit='steps')
    for step, (images, _, masks, lab) in enumerate(pbar):
        with torch.no_grad():
            images = images.to(device, non_blocking=True)
            lab = lab.to(device, non_blocking=True)
            masks = masks.squeeze(1).to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                modals = modal_extractor(images)

                images_norm = TF.normalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                inp = [images_norm] + modals

                anomaly, confidence, detection = model(inp)

                val_loss = criterion(anomaly, masks, confidence, detection, lab)
            val_loss_avg.update(val_loss.detach().item())
            scores.append(detection.squeeze().detach().cpu().item())
            labels.append(lab.squeeze().detach().cpu().item())

    auc, baCC = computeDetectionMetrics(scores, labels)
    writer.add_scalar('Val Loss', val_loss_avg.average(), epoch)
    writer.add_scalar('Val AUC', auc, epoch)
    writer.add_scalar('Val bACC', baCC, epoch)
    if val_loss_avg.average() < min_loss:
        min_loss = val_loss_avg.average()
        result = {'epoch': epoch, 'val_loss': val_loss_avg.average(),'val_baCC': baCC,
                  'val_auc': auc, 'state_dict': model.state_dict(),
                  'extractor_state_dict': modal_extractor.state_dict()}
        torch.save(result, './ckpt/{}/best_val_loss.pth'.format(config.MODEL.NAME))

result = {'epoch': config.EPOCHS - 1, 'val_loss': val_loss_avg.average(), 'val_baCC': baCC,
                  'val_auc': auc, 'state_dict': model.state_dict(),
          'extractor_state_dict': modal_extractor.state_dict()}
torch.save(result, './ckpt/{}/final.pth'.format(config.MODEL.NAME))
