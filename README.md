## Exploring Multi-Modal Fusion for Image Manipulation Detection and Localization (MMM 2024)

Official implementation of the MMM 2024 paper : "Exploring Multi-Modal Fusion for Image Manipulation Detection and Localization"


## Datasets


### Data folder structure
Prepare all train datasets:
- [Casiav2](https://github.com/namtpham/casia2groundtruth)
- [tampCOCO](https://github.com/mjkwon2021/CAT-Net#1-downloading-tampcoco--compraise)
- [IMD2020](http://staff.utia.cas.cz/novozada/db/)
- [FantasticReality](http://zefirus.org/articles/9f78c1e9-8652-4392-9199-df1b6a6c1a3d/)

Prepare all test datasets:
- [Casiav1](https://github.com/namtpham/casia1groundtruth)
- [corel](https://www.kaggle.com/datasets/elkamel/corel-images)[^1]
- [CocoGlide](https://github.com/grip-unina/TruFor#cocoglide-dataset)
- [Columbia](https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/)
- [COVER](https://github.com/wenbihan/coverage)
- [DSO-1](https://recodbr.wordpress.com/code-n-data/#dso1_dsi1)

[^1] The corel dataset is needed to create the Casiav1+ dataset.

Then all datasets are structured as:
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

