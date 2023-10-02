## Exploring Multi-Modal Fusion for Image Manipulation Detection and Localization (MMM 2024)

Official implementation of the MMM 2024 paper : "Exploring Multi-Modal Fusion for Image Manipulation Detection and Localization"


## Datasets

Download train datasets:
- [Casiav2](https://github.com/namtpham/casia2groundtruth)
- [tampCOCO](https://github.com/mjkwon2021/CAT-Net#1-downloading-tampcoco--compraise)
- [IMD2020](http://staff.utia.cas.cz/novozada/db/)
- [FantasticReality](http://zefirus.org/articles/9f78c1e9-8652-4392-9199-df1b6a6c1a3d/)

Download test datasets:
- [Casiav1](https://github.com/namtpham/casia1groundtruth)
- [corel](https://www.kaggle.com/datasets/elkamel/corel-images)
- [CocoGlide](https://github.com/grip-unina/TruFor#cocoglide-dataset)
- [Columbia](https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/)
- [COVER](https://github.com/wenbihan/coverage)
- [DSO-1](https://recodbr.wordpress.com/code-n-data/#dso1_dsi1)

The corel dataset is needed to create the Casiav1+ dataset.

### Data folder structure

Then you should place the datasets in the data directory as such:
```
data/
├── Casiav1
│   ├── Au
│   ├── Gt
│   └── Tp
├── Casiav2
│   ├── Au
│   ├── mask
│   └── tampered
├── CocoGlide
│   ├── fake
│   ├── mask
│   └── real
├── Columbia
│   ├── 4cam_auth
│   └── 4cam_splc
├── compRAISE
│   └── <all images here>
├── corel-1k
│   ├── test_set
│   └── training_set
├── COVER
│   ├── Au
│   ├── mask
│   └── tampered
├── DSO-1
│   ├── images
│   └── masks
├── FantasticReality
│   ├── ColorFakeImages
│   ├── ColorRealImages
│   └── masks
├── IMD2020
│   ├── 1a1ogs
│   ├── 1a3oag
│       .
│       .
│       .
│   └── z41
└── tampCOCO
    └── <all images here>
```

## Training
### Preparation
Before training, you need to download the pretrained networks following the instructions [here](pretrained/README.md) and place them in the <root>/pretrained directory as:
```
pretrained/
├── segformer
├── noiseprint
└── modal_extractor
```
### Localization Training
After that you can run an example training by:

```bash
example_train.sh
```
You can change the training parameters by creating a new experiment yaml file in the <root>/experiments directory.
The checkpoint is saved as:
```
ckpt/
└── <model_name>
    ├── best_val_loss.pth
    └── final.pth
```
The model_name parameter is set in the experiment yaml file as:
```yaml
MODEL:
  NAME: <model_name>
```
### Detection Training
To run detection training (phase 2) you need a localization checkpoint (produced from phase 1 training) placed in the chekpoints folder. 
If you want to use one of our localization checkpoints you can download them following the instructions [here](ckpt/README.md).

The experiment file for phase 2 training should be set for detection as:
```yaml
MODEL:
  TRAIN_PHASE: 'detection'
```
Then you can train a model for detection and evaluate on our testing datasets as follows:
```bash
source data.sh
exp='./experiments/ec_example_phase2.yaml'
ckpt_loc='./ckpt/<path_to_localization_ckpt>'
ckpt='./ckpt/<model_name>/best_val_loss.pth'
$pint ec_train_phase2.py --ckpt $ckpt_loc --exp $exp

$pint test_detection.py --exp $exp --ckpt $ckpt --manip $columbia_manip --auth $columbia_auth
$pint test_detection.py --exp $exp --ckpt $ckpt --manip $cover_manip --auth $cover_auth
$pint test_detection.py --exp $exp --ckpt $ckpt --manip $dso1_manip --auth $dso1_auth
$pint test_detection.py --exp $exp --ckpt $ckpt --manip $cocoglide_manip --auth $cocoglide_auth
$pint test_detection.py --exp $exp --ckpt $ckpt --manip $casiav1_manip --auth $casiav1_auth
```

## Evaluation
You can download our pretrained networks following the instructions [here](ckpt/README.md) and place them in the <root>/ckpt directory.
Then you can evaluate a model by using:
```bash
example_test.sh
```
and changing the relevant parameters.

## Acknowledgements
Thanks to the public repositories:
- [DELIVER](https://github.com/jamycheung/DELIVER)
- [TruFor](https://github.com/grip-unina/TruFor)
- [CAT-Net](https://github.com/mjkwon2021/CAT-Net)

## Citation
If you use some resources provided by this repo, please cite this paper.

* Exploring Multi-Modal Fusion for Image Manipulation Detection and Localization
````
@inproceedings{triaridis2023exploring
    title={Exploring Multi-Modal Fusion for Image Manipulation Detection and Localization},
    author={Triaridis, Konstantinos and Mezaris, Vasileios},
    year={2023}
}
````
