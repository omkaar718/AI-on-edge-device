import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm 
#from skimage.io import imread
from PIL import Image
import argparse

from utils import load_data

def test(model, threshold, X_test, Y_test):
    iou_metric = tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=threshold)
    iou_metric.reset_state()

    preds = model.predict(X_test)
    #print(preds)
    iou_metric.update_state(Y_test, preds)
    print('IoU Score : ',iou_metric.result().numpy())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_width", default=512, type=int)
    parser.add_argument("--img_height", default=512, type=int)
    parser.add_argument("--img_channels", default=3, type=int)
    parser.add_argument("--mask_height", default=16, type=int)
    parser.add_argument("--mask_width", default=16, type=int)
    parser.add_argument("--data_path", 
                        default='/data/oprabhune/tianen_colab/segmentation_approach/segmentation_dataset/', type=str)
    parser.add_argument("--thresh", default=0.5, type=float)
    parser.add_argument("--model_path", default='saved_during_training', type=str)
    args = parser.parse_args()

    #Data Loading
    X_test, Y_test = load_data(
        args.data_path,
        'test', 
        (args.img_width, args.img_height, args.img_channels), 
        (args.mask_width, args.mask_height)
    )

    # Load model
    #model = tf.keras.models.load_model('saved_during_training_alpha_0.25')
    model = tf.keras.models.load_model(args.model_path)

    # test
    test(model, args.thresh, X_test, Y_test)

if __name__ == "__main__":
    main()