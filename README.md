
# Solving a problem of MDE
Source: Keras Docs

# Dataset: DIODE (validation set)
- Given dataset format : (image.png, image_depth.npy, image_depth_mask.npy)
- Shape (image.png): (768, 1024, 3)
- Shape (image_depth.npy): (768, 1024, 1) - max 225.18, min 0.0,
- Shape (image_depth_mask.npy): (768, 1024) - contain 0 or 1, where 1 means the pixel is valid and 0 means invalid

# Dataset Directory Structure

```
val_extracted
└── val
    ├── indoors
    │   ├── scene_00019
    │   │   └── scan_00183
    │   ├── scene_00020
    │   │   ├── scan_00184
    │   │   ├── scan_00185
    │   │   ├── scan_00186
    │   │   └── scan_00187
    │   └── scene_00021
    │       ├── scan_00188
    │       ├── scan_00189
    │       ├── scan_00190
    │       ├── scan_00191
    │       └── scan_00192
    └── outdoor
        ├── scene_00022
        │   ├── scan_00193
        │   ├── scan_00194
        │   ├── scan_00195
        │   ├── scan_00196
        │   └── scan_00197
        ├── scene_00023
        │   ├── scan_00198
        │   ├── scan_00199
        │   └── scan_00200
        └── scene_00024
            ├── scan_00201
            └── scan_00202
```
Total Images: 774

