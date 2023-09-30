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
### Detection Training

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
