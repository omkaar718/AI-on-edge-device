import tensorflow as tf
import numpy as np
import cv2
import os
import math


dir = 'data/val/images/'
new_model = tf.keras.models.load_model('./smaller_model_saved_during_training_alpha_0.25/')

files = os.listdir(dir)


for img_path in files:
    #img_path = "test_images/sample_3_orig.jpg"
    #print(img_path)
    orig_img = cv2.imread(dir + img_path)
    orig_image_size = orig_img.shape
    #print(orig_image_size)
    image_size = 224 # 512x512
    

    img = cv2.resize(orig_img, (image_size, image_size))
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_array = np.expand_dims(img,axis=0).astype(np.float32)
    # Predict
    predictions = tf.squeeze(new_model.predict(img_array))
    resolution_scaling_factor = 0.5

    #print(img_2.shape)
    '''
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


    img_2 = cv2.resize(img, None, fx = resolution_scaling_factor, fy = resolution_scaling_factor)
    img_2 = cv2.resize(img_2, (orig_image_size[1], orig_image_size[0]))
    '''

    grid_size = predictions.shape[0]
    grid_w, grid_h = (math.ceil(orig_image_size[1]/grid_size), math.ceil(orig_image_size[0]/grid_size))
    #print(grid_w, grid_h)
    
    
    for threshold in range(50, 100, 20):
        #img_2 = np.zeros(shape = orig_image_size)
        img_2 = cv2.resize(orig_img, None, fx = resolution_scaling_factor, fy = resolution_scaling_factor)
        img_2 = cv2.resize(img_2, (orig_image_size[1], orig_image_size[0]))                                 

        threshold = round(0.01*threshold, 1)
        #print(threshold)

        roi_indices = np.argwhere(predictions > threshold)

        for i in roi_indices:
            row = grid_h*i[0]
            col = grid_w*i[1]
            img_2[row: row + grid_h, col: col + grid_w] = orig_img[row: row + grid_h, col: col + grid_w]
        
        folder = f'data/val/sf_{resolution_scaling_factor}_reconstructed_th_{threshold}'
        if not os.path.isdir(folder):
            os.mkdir(folder)
        cv2.imwrite(f'{folder}/{img_path}', img_2)

print('Done\n')
