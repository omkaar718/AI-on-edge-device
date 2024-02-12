import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm 
#from skimage.io import imread
from PIL import Image
import argparse
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from utils import load_data

def test(interpreter_quant, input_index, output_index, threshold, X_test, Y_test):
    iou_metric = tf.keras.metrics.MeanIoU(num_classes=2)
    #iou_metric.reset_state()
    preds = []
    #print(preds.shape)
    #preds = model.predict(X_test)
    for data_index, x in enumerate(X_test):
        x = np.expand_dims(x,axis=0).astype(np.float32)
        interpreter_quant.set_tensor(input_index, x)
        interpreter_quant.invoke()
        predictions = interpreter_quant.get_tensor(output_index)
        #print(predictions.shape)
        #print(preds[0].shape)
        preds.append(predictions)
    #print(predictions)
    preds = np.array(preds)
    with open(f'Y_test.npy', 'wb') as f:
        np.save(f, Y_test.astype(int))

    with open(f'preds.npy', 'wb') as f:
        np.save(f, preds)

    return
    preds = preds > threshold
    dim = np.prod(Y_test.shape)
    print(classification_report(Y_test.astype(int).reshape(dim), preds.astype(int).reshape(dim)))
    return


    #print(preds)
    #iou_metric.update_state(Y_test, preds)
    #print('IoU Score : ',iou_metric.result().numpy())
    #print('Accuracy: ', np.mean(np.equal(preds, Y_test)))
    print(type(Y_test), type(preds))
    print(type(Y_test.astype(int)), type(preds.astype(int)))
    
    with open(f'Y_test.npy', 'wb') as f:
        np.save(f, Y_test)

    with open(f'preds.npy', 'wb') as f:
        np.save(f, preds)
    #print(classification_report(Y_test.astype(int), preds.astype(int)))
    print('precision : ', precision_score)
    print('recall: ', recall_score)
    print('f1 : ', f1_score)
    print('\n\n')

def calc_scores(Y_test, preds, thresh):
    preds_ = preds > thresh
    
    dim = np.prod(Y_test.shape)
    print(classification_report(Y_test.reshape(dim), preds_.astype(int).reshape(dim)))


'''
def calc_scores():
    preds = np.load('preds.npy')
    Y_test = np.load('Y_test.npy')
    preds = preds.astype(int)
    Y_test = Y_test.astype(int)
    predicted_masks = np.squeeze(preds)
    true_masks = np.squeeze(Y_test)
    print(predicted_masks.shape, true_masks.shape)
    predictions_flat = tf.reshape(predicted_masks, [-1])
    ground_truth_flat = tf.reshape(true_masks, [-1])

    # Calculate accuracy
    accuracy, accuracy_update_op = tf.metrics.accuracy(ground_truth_flat, predictions_flat)

    # Calculate precision
    precision, precision_update_op = tf.metrics.precision(ground_truth_flat, predictions_flat)

    # Calculate recall
    recall, recall_update_op = tf.metrics.recall(ground_truth_flat, predictions_flat)

    # F1 score (requires precision and recall)
    f1_score = 2 * (precision * recall) / (precision + recall)

    print(accuracy, precision, recall, f1_score)
'''


def load_tflite_model(tflite_model_path):
    interpreter_quant = tf.lite.Interpreter(model_path=str(tflite_model_path))
    interpreter_quant.allocate_tensors()
    input_index = interpreter_quant.get_input_details()[0]["index"]
    output_index = interpreter_quant.get_output_details()[0]["index"]
    return interpreter_quant, input_index, output_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_width", default=240, type=int)
    parser.add_argument("--img_height", default=240, type=int)
    parser.add_argument("--img_channels", default=3, type=int)
    parser.add_argument("--mask_height", default=15, type=int)
    parser.add_argument("--mask_width", default=15, type=int)
    parser.add_argument("--data_path", 
                        default='/storage/oprabhune/tianen_colab/face_roi/face_roi_dataset/', type=str)
    parser.add_argument("--thresh", default=0.5, type=float)
    parser.add_argument("--model_path", default='face_roi_alpha_0.1_240_15', type=str)
    args = parser.parse_args()

    # Load model
    #model = tf.keras.models.load_model('saved_during_training_alpha_0.25')
    #model = tf.keras.models.load_model(args.model_path)

    interpreter_quant, input_index, output_index = load_tflite_model('dynamic_face_roi_alpha_240_15.tflite')

    '''
    #Data Loading
    X_test, Y_test = load_data(
        args.data_path,
        'test', 
        (args.img_width, args.img_height, args.img_channels), 
        (args.mask_width, args.mask_height)
    )
    '''
   

    # test
    for thresh in np.arange(0, 1.1, 0.1):
        preds = np.load('preds.npy')
        Y_test = np.load('Y_test.npy')
        print("\nThreshold : ", thresh)
        #test(interpreter_quant, input_index, output_index, thresh, X_test, Y_test)
        
        calc_scores(Y_test, preds, thresh)

if __name__ == "__main__":
    main()
