'''
import tensorflow as tf
import tensorflow_datasets as tfds
import argparse
import numpy as np
import pathlib

from PIL import Image
from collections import namedtuple

parser = argparse.ArgumentParser(description="Convert a TF SavedModel to a TFLite model")
#parser.add_argument("--model-name", default
#                    help="Name of the model. See tools/train_model.sh for semantics of model name")
parser.add_argument("--dataset",  default="coco2014",
                    help="Name of the TFRecord dataset that should be used for quantization") 
parser.add_argument("--num-samples", help="Number of samples to calibrate on", type=int, default=100)
parser.add_argument("--input-height", type=int, default=240)
parser.add_argument("--input-width", type=int, default=240)

ImgShape = namedtuple('ImageShape', 'height width channels')

def fake_data_gen(num_samples, input_shape):
    def representative_dataset_gen():
        for i in range(num_samples):
            yield [np.ones((-1, input_shape.height, input_shape.width, input_shape.channels)).astype(np.float32)]
    return representative_dataset_gen

def make_data_gen(dataset_name, num_samples, input_shape):
    """
    Uses the images from datadir/Images to quantize the model. 
    """
    if dataset_name == "fake":
        return fake_data_gen(num_samples, input_shape)

    datadir = pathlib.Path('data/raw') / dataset_name
    imgdir = datadir / 'Images'
    imgdir = pathlib.Path('train_images')
    def representative_dataset_gen():
        for i, filename in enumerate(imgdir.iterdir()):
            if filename.suffix not in ['.jpeg', '.jpg', '.png']: 
                continue
            image = Image.open(str(filename.resolve()))
            image = image.resize((input_shape.height, input_shape.width))
            yield [np.array(image).reshape(-1, input_shape.height, input_shape.width, input_shape.channels).astype(np.float32)]
            if i >= num_samples: break
    return representative_dataset_gen

def main():
    args = parser.parse_args()
    input_shape = ImgShape(height=args.input_height, width=args.input_width, channels=3)
    #model_savedir = f'exported_models/{args.model_name}/saved_model'
    model_savedir = './stripped_clustered_model'
    converter = tf.lite.TFLiteConverter.from_saved_model(model_savedir, signature_keys=['serving_default'])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = make_data_gen(args.dataset, args.num_samples, input_shape)
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to uint8 (APIs added in r2.3)
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    quantized_model = converter.convert()
    bytes = open(f'pruned_model.tflite', "wb").write(quantized_model)

    
if __name__ == "__main__":
    main()

'''


import tensorflow as tf
import argparse
import numpy as np
import pathlib

from PIL import Image
from collections import namedtuple

parser = argparse.ArgumentParser(description="Convert a TF SavedModel to a TFLite model")
parser.add_argument("--model-name", 
                    help="Name of the model. See tools/train_model.sh for semantics of model name")
parser.add_argument("--dataset",  default="coco2014",
                    help="Name of the TFRecord dataset that should be used for quantization") 
parser.add_argument("--num-samples", help="Number of samples to calibrate on", type=int, default=1000)
parser.add_argument("--input-height", type=int, default=240)
parser.add_argument("--input-width", type=int, default=240)

ImgShape = namedtuple('ImageShape', 'height width channels')

def fake_data_gen(num_samples, input_shape):
    def representative_dataset_gen():
        for i in range(num_samples):
            yield [np.ones((-1, input_shape.height, input_shape.width, input_shape.channels)).astype(np.float32)]
    return representative_dataset_gen

def make_data_gen(dataset_name, num_samples, input_shape):
    """
    Uses the images from datadir/Images to quantize the model. 
    """
    if dataset_name == "fake":
        return fake_data_gen(num_samples, input_shape)

    
    #datadir = pathlib.Path('data/raw') / dataset_name
    #imgdir = datadir / 'Images'
    imgdir = pathlib.Path('face_roi_dataset/face_roi_dataset/train_images')
    def representative_dataset_gen():
        for i, filename in enumerate(imgdir.iterdir()):
            if filename.suffix not in ['.jpeg', '.jpg', '.png']: 
                continue
            image = Image.open(str(filename.resolve())).convert("RGB")
            #print('\n\n', i, filename, image.size)
            image = image.resize((input_shape.height, input_shape.width))
            #print(image.size)
            yield [np.array(image).reshape(-1, input_shape.height, input_shape.width, input_shape.channels).astype(np.float32)]
            if i >= num_samples: break
    return representative_dataset_gen

def main():
    args = parser.parse_args()
    input_shape = ImgShape(height=args.input_height, width=args.input_width, channels=3)
    model_savedir = 'face_roi_alpha_0.1_240_15'

    converter = tf.lite.TFLiteConverter.from_saved_model(model_savedir, signature_keys=['serving_default'])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    #converter.representative_dataset = make_data_gen(args.dataset, args.num_samples, input_shape)
    # Ensure that if any ops can't be quantized, the converter throws an error
    #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to uint8 (APIs added in r2.3)
    #converter.inference_input_type = tf.int8
    #converter.inference_output_type = tf.int8
    quantized_model = converter.convert()
    bytes = open(f'dynamic_face_roi_alpha_240_15.tflite', "wb").write(quantized_model)


    
if __name__ == "__main__":
    main()
