import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
#from model import UNET
#from cnn_backbone_custom import cnn_backbone_custom
from torchvision import models
#from MobileNetV3WithConv import MobileNetV3WithConv
from EfficientNetWithConv import EfficientNetWithConv
torch.manual_seed(1)
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
NUM_EPOCHS = 50
NUM_WORKERS = 4
IMAGE_HEIGHT = 512  # 1280 originally
IMAGE_WIDTH = 512  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
'''
TRAIN_IMG_DIR = "./data/train_images/"
TRAIN_MASK_DIR = "./data/train_masks/"
VAL_IMG_DIR = "./data/val_images/"
VAL_MASK_DIR = "./data/val_masks/"
'''
TRAIN_IMG_DIR = "/data/oprabhune/tianen_colab/segmentation_approach/segmentation_dataset/train_images/"
TRAIN_MASK_DIR = "/data/oprabhune/tianen_colab/segmentation_approach/segmentation_dataset/train_masks/"
VAL_IMG_DIR = "/data/oprabhune/tianen_colab/segmentation_approach/segmentation_dataset/val_images/"
VAL_MASK_DIR = "/data/oprabhune/tianen_colab/segmentation_approach/segmentation_dataset/val_masks/"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
    
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            #print('Shape of prediction: ', predictions.size())
            #predictions = torch.sum(predictions, 1).unsqueeze(1)

            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    # LOAD MODEL
    '''
    #model = cnn_backbone_custom().to(DEVICE)
    model = models.mobilenet_v3_large()
    num_classes = 2
    model.classifier = nn.Sequential(
            nn.Linear(in_features=960, out_features=1280, bias=True),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        )
    model = model.to(DEVICE)
    model.load_state_dict(torch.load('best_weights_mobilenet_classifier.pt', map_location=DEVICE))
    model = model.features
    '''
    model = EfficientNetWithConv().to(DEVICE)


    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


    current_acc, best_dice_score = check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        print('\nEpoch : ', epoch)
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        _, current_dice_score =  check_accuracy(val_loader, model, device=DEVICE)
        if(current_dice_score > best_dice_score):
            print('Saving checkpoint')
            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)
            save_predictions_as_imgs(val_loader, model, folder="saved_images/", device=DEVICE)
            best_dice_score = current_dice_score
        '''
        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )
        '''


if __name__ == "__main__":
    main()
