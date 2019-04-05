Anime Sketch Coloring with Swish-Gated Residual U-Net
=====================================================

Authors: _Gang Liu, Xin Chen, Yanzhong Hu_

Using Deep Learning to colorize manga based on paper with same name

Setup
-----

Use the `requirements.txt` file to install the necessary depedencies for this
project.

```
$ pip install -r requirements.txt
```

Before proceeding, make sure the data folder has the following structure.
```
data/
├── images/
│   ├── images_bw/
│   └── images_rgb/
└── vgg_19.ckpt
```

Training the SGRU Model
-----------------------

```
$ ./train.py ${DATA_DIR} ${OUTPUT_DIR}
```

The output directory will have the following structure
```
${OUTPUT_DIR}/
└── ${EXP_NAME:-TIME_STAMP}/
    ├── images/
    ├── events.out.tfevents (tensorboard)
    └── model.ckpt
```

Preliminary Results
-------------------
We are experiencing an issue where the generated images are washed out, or have a "watercolor" effect, and not as vibrant as the results in the original paper. The following image shows this (first column is the sketch input, second column is the original image, and the rest are the generated images):
![Results](https://i.imgur.com/QDhX33A.jpg)
If you see any issues with our implementation, please create a pull request or an issue. Thanks!