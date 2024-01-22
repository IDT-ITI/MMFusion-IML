import argparse
import numpy as np
from torch.utils.data import DataLoader
import torch
import torchvision.transforms.functional as TF
import logging
import matplotlib.pyplot as plt
from data.datasets import ManipulationDataset
from models.cmnext_conf import CMNeXtWithConf
from models.modal_extract import ModalitiesExtractor
from configs.cmnext_init_cfg import _C as config, update_config

parser = argparse.ArgumentParser(description='Infer')
parser.add_argument('-gpu', '--gpu', type=int, default=0, help='device, use -1 for cpu')
parser.add_argument('-log', '--log', type=str, default='INFO', help='logging level')
parser.add_argument('-exp', '--exp', type=str, default='experiments/ec_example_phase2.yaml', help='Yaml experiment file')
parser.add_argument('-ckpt', '--ckpt', type=str, default='ckpt/early_fusion_detection.pth', help='Checkpoint')
parser.add_argument('-path', '--path', type=str, default='example.png', help='Image path')
parser.add_argument('opts', help="other options", default=None, nargs=argparse.REMAINDER)

args = parser.parse_args()

config = update_config(config, args.exp)

loglvl = getattr(logging, args.log.upper())
logging.basicConfig(level=loglvl)

gpu = args.gpu

device = 'cuda:%d' % gpu if gpu >= 0 else 'cpu'
np.set_printoptions(formatter={'float': '{: 7.3f}'.format})

if device != 'cpu':
    # cudnn setting
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = False
    cudnn.deterministic = True
    cudnn.enabled = config.CUDNN.ENABLED


modal_extractor = ModalitiesExtractor(config.MODEL.MODALS[1:], config.MODEL.NP_WEIGHTS)

model = CMNeXtWithConf(config.MODEL)

ckpt = torch.load(args.ckpt)

model.load_state_dict(ckpt['state_dict'])
modal_extractor.load_state_dict(ckpt['extractor_state_dict'])

modal_extractor.to(device)
model = model.to(device)
modal_extractor.eval()
model.eval()

target = args.path.split(".")[-2] + "_mask.png"

with open('tmp_inf.txt', 'w') as f:
    f.write(args.path + ' None 0\n')

val = ManipulationDataset('tmp_inf.txt',
                          config.DATASET.IMG_SIZE,
                          train=False)
val_loader = DataLoader(val,
                        batch_size=1,
                        shuffle=False,
                        num_workers=config.WORKERS,
                        pin_memory=True)

f1 = []
f1th = []
for step, (images, _, masks, lab) in enumerate(val_loader):
    with torch.no_grad():
        images = images.to(device, non_blocking=True)
        masks = masks.squeeze(1).to(device, non_blocking=True)

        modals = modal_extractor(images)

        images_norm = TF.normalize(images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        inp = [images_norm] + modals

        anomaly, confidence, detection = model(inp)

        gt = masks.squeeze().cpu().numpy()
        map = torch.nn.functional.softmax(anomaly, dim=1)[:, 1, :, :].squeeze().cpu().numpy()
        det = detection.item()

        plt.imsave(target, map, cmap='RdBu_r', vmin=0, vmax=1)

print(f"Ran on {args.path}")
print(f"Detection score: {det}")
print(f"Localization map saved in {target}")