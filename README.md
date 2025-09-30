# FEGNet

Official implementation of **FEG-Net** for medical image segmentation.

## Dataset

Organize your data as follows:
```
data/
├── train/
│   ├── images/   # e.g., .jpg or .png
│   └── masks/    # corresponding binary masks (.png, 0 for background, 255 for foreground)
└── test/
    ├── images/
    └── masks/    # (optional, for evaluation)
```
> Image and mask files must have the same name (e.g., `001.jpg` ↔ `001.png`).

## Training

```bash
python train.py --data_path ./data
```
The best model will be saved in `./checkpoints/best_model.pth`.

## Testing

```bash
python test.py --data_path ./data --model_path ./checkpoints/best_model.pth
```
Predictions will be saved in `./results/`.
```
