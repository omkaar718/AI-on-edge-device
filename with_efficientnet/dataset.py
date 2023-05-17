import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as transforms
#import albumentations as A
import torch
class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        self.images = os.listdir(image_dir)
        self.images =  [i for i in self.images if '.jpg' in i]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        #check_unique = np.array(Image.open(mask_path))
        #print('\nUnique values in check_uniqueue: ', np.unique(check_unique))
        #mask[mask == 255.0] = 1.0
        mask[mask < 127] = 0.0
        mask[mask >= 127] = 1.0


        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
            #print('Unique values in mask after augmentations: ', np.unique(mask, return_counts = True))
            #Image.fromarray(mask.cpu().numpy()).save('aumented.jpg')
        #print('Mask ',type(mask), mask.size())
        #mask = A.Resize((64, 64))(image = mask)['image']
        #mask = transforms.Resize((64, 64))(torch.unsqueeze(mask, dim=0))
        # For mobilenet backbone with input dim 512x512
        mask = transforms.Resize((16, 16), antialias=True)(torch.unsqueeze(mask, dim=0))

        mask = torch.squeeze(mask, dim = 0)
        return image, mask

