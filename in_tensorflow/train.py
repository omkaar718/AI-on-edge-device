import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm 
#from skimage.io import imread
from PIL import Image
import argparse

def load_data(TRAIN_PATH, input_shape, mask_shape):
    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = input_shape
    mask_height, mask_width = mask_shape
    train_ids = next(os.walk(TRAIN_PATH + 'train_images/'))[2]
    #val_ids = next(os.walk(VAL_PATH + 'images/'))[2]
    
    # Train data
    #test_ids = next(os.walk(TEST_PATH))[1]
    X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Y_train = np.zeros((len(train_ids), mask_height, mask_width, 1), dtype=bool)

    print('Resizing training images and masks')
    for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):   
        if('jpg' not in id_):
            continue
        img_path = TRAIN_PATH + 'train_images/' + id_
        #img = imread(img_path)
        img = np.array(Image.open(img_path).convert("RGB"))
        #print(img.shape, img_path)
        #img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
        img = tf.image.resize_with_pad(img, IMG_HEIGHT, IMG_WIDTH)
        X_train[n] = img  #Fill empty X_train with values from img

        mask_path = TRAIN_PATH + 'train_masks/' + id_
        #mask_ = imread(mask_path)
        mask_ = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask_ = tf.image.resize_with_pad(mask_[ :, :, np.newaxis], mask_height, mask_width, method=tf.image.ResizeMethod.AREA, antialias=False)
        #print(type(mask_), mask_.max(), mask_.shape, mask_)
        #mask_[mask_ > 0] = 1
        mask_ = mask_ > 0
        #print(mask_)
        Y_train[n] = mask_

    return X_train, Y_train

def load_model(input_shape):
    # when include_preprocessing = True, keep input unnormalized i.e. 0 - 255
    model_1 = tf.keras.applications.MobileNetV3Small(
        input_shape=input_shape,
        alpha=1.0,
        minimalistic=False,
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        pooling=None,
        dropout_rate=0.2,
        include_preprocessing = True
    )
    output = tf.keras.layers.Conv2D(
        filters=1, 
        kernel_size=3, 
        strides=1, 
        padding='same',
        activation='sigmoid')(model_1.layers[-1].output)
    model = tf.keras.Model(inputs=model_1.inputs, outputs=output)
    return model

def train(model, X_train, Y_train):
    # train
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    callback_checkpointer = tf.keras.callbacks.ModelCheckpoint(
        'saved_during_training', 
        verbose=1, 
        save_best_only=True, 
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max')

    callback_tf_logger = tf.keras.callbacks.TensorBoard(log_dir='logs_tf_logger')
    results = model.fit(X_train, Y_train, validation_split=0.2, batch_size=32, epochs=30, callbacks=[callback_checkpointer, callback_tf_logger])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_width", default=512, type=int)
    parser.add_argument("--img_height", default=512, type=int)
    parser.add_argument("--img_channels", default=3, type=int)
    parser.add_argument("--mask_height", default=16, type=int)
    parser.add_argument("--mask_width", default=16, type=int)
    parser.add_argument("--train_data_path", 
                        default='/data/oprabhune/tianen_colab/segmentation_approach/segmentation_dataset/', type=str)

    args = parser.parse_args()

    #Data Loading
    X_train, Y_train = load_data(
        args.train_data_path, 
        (args.img_width, args.img_height, args.img_channels), 
        (args.mask_width, args.mask_height)
    )

    # Load model
    model = load_model(input_shape = (args.img_width, args.img_height, args.img_channels))

    # train
    train(model, X_train, Y_train)

if __name__ == "__main__":
    main()

