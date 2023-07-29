import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm 
#from skimage.io import imread
from PIL import Image
import argparse
import larq_zoo as lqz
import larq as lq
from utils import custom_data_loader


def load_model(input_shape):
    
    model = tf.keras.models.Sequential([
            lqz.sota.QuickNet(input_shape = input_shape, weights="imagenet", include_top=False),
            lq.layers.QuantConv2D(
                filters=1, 
                kernel_size=3, 
                strides=1, 
                padding='same',
                input_quantizer="ste_sign",
                kernel_quantizer="ste_sign",
                kernel_constraint="weight_clip",
                use_bias=False,
                activation='sigmoid')
            ])
    print('\n', lq.models.summary(model))
    return model



def train(model, train_dataset, val_dataset, batch_size, epochs):
        # train
        opt = tf.keras.optimizers.Adam(lr=0.01, decay=0.001)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
        callback_checkpointer = tf.keras.callbacks.ModelCheckpoint(
            'saved_during_training_quicknet', 
            verbose=1, 
            save_best_only=True, 
            save_weights_only=False,
            monitor='val_mean_io_u',
            mode='max')

        callback_tf_logger = tf.keras.callbacks.TensorBoard(log_dir='logs_tf_logger')
        results = model.fit(train_dataset, validation_data=val_dataset, batch_size=batch_size, epochs=epochs, callbacks=[callback_checkpointer, callback_tf_logger])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_width", default=512, type=int)
    parser.add_argument("--img_height", default=512, type=int)
    parser.add_argument("--img_channels", default=3, type=int)
    parser.add_argument("--mask_height", default=16, type=int)
    parser.add_argument("--mask_width", default=16, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--data_mode", default='train', type=str)
    parser.add_argument("--train_data_path", 
                        default='/storage/oprabhune/tianen_colab/segmentation_approach/segmentation_dataset/', type=str)

    args = parser.parse_args()
    
    '''
    #Data Loading
    X_train, Y_train = load_data(
        args.train_data_path, 
        'train',
        (args.img_width, args.img_height, args.img_channels), 
        (args.mask_width, args.mask_height)
    )
    np.save('X_train_512_scaled.npy', X_train)
    np.save('Y_train_16.npy', Y_train)
    '''
    # Load model
    model = load_model(input_shape = (args.img_width, args.img_height, args.img_channels))
    
    # Load data
    if(args.data_mode == 'train'):
         val_split = 0.2
    else:
         val_split = 0

    train_dataset, val_dataset = custom_data_loader(
         args.train_data_path, 
         args.batch_size,
         (args.img_width, args.img_height),
         (args.mask_width, args.mask_height),
         args.data_mode,
         val_split

    )

    
    # train
    train(model, train_dataset, val_dataset, args.batch_size, args.epochs)

if __name__ == "__main__":
    main()
