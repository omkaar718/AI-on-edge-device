import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm 
#from skimage.io import imread
from PIL import Image

def load_data(TRAIN_PATH, data_mode, input_shape, mask_shape):
    print('\n\nData path : ', TRAIN_PATH + f'{data_mode}_images/')

    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = input_shape
    mask_height, mask_width = mask_shape
    train_ids = next(os.walk(TRAIN_PATH + f'{data_mode}_images/'))[2]
    print(type(train_ids))
    #train_ids = train_ids[:1000]
    #val_ids = next(os.walk(VAL_PATH + 'images/'))[2]
    
    # Train data
    #test_ids = next(os.walk(TEST_PATH))[1]
    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), mask_height, mask_width, 1), dtype=bool)

    print('Resizing training images and masks')
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
        if('jpg' not in id_):
            continue
        img_path = TRAIN_PATH + f'{data_mode}_images/' + id_
        #img = imread(img_path)
        img = np.array(Image.open(img_path).convert("RGB"))
        #print(img.shape, img_path)
        #img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        img = tf.image.resize_with_pad(img, IMG_HEIGHT, IMG_WIDTH)
        X_train[n] = img  #Fill empty X_train with values from img

        mask_path = TRAIN_PATH + f'{data_mode}_masks/' + id_
        #mask_ = imread(mask_path)
        mask_ = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask_ = tf.image.resize_with_pad(mask_[ :, :, np.newaxis], mask_height, mask_width, method=tf.image.ResizeMethod.AREA, antialias=False)
        #print(type(mask_), mask_.max(), mask_.shape, mask_)
        #mask_[mask_ > 0] = 1
        mask_ = mask_ > 0
        #print(mask_)
        Y_train[n] = mask_

    return X_train, Y_train

if __name__ == "__main__":
    load_data('/storage/oprabhune/tianen_colab/segmentation_approach/segmentation_dataset/',
            'train',
            (256, 256, 3),
            (16, 16)
            )
