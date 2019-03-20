Anime-Sketch-Coloring-with-Swish-Gated-Residual-UNET
====================================================

Using Deep Learning to colorize manga based on paper with same name

Setup
-----

(Talk about prerequisites here)

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
