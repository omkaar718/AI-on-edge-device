import tensorflow as tf
import csv
# opening the CSV file
import requests
import cv2
import os
import numpy as np
from PIL import Image
import math
from deepface import DeepFace
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse

def resize_with_pad(image, new_shape, padding_color = (0, 0, 0)):
    """Maintains aspect ratio and resizes with padding.
    Params:
        image: Image to be resized.
        new_shape: Expected (width, height) of new image.
        padding_color: Tuple in BGR of padding color
    Returns:
        image: Resized image with padding
    """
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=padding_color)
    return image, ratio, top, left


def getHeightWidth(hw):
    '''get raw height and width of image'''
    hw = hw.split(' ')
    hw = [int(p) for p in hw]
    return hw

def getCoordinate(rect):
    '''get face coordinates in format of xmin, ymin, xmax, ymax'''
    rect = rect.split(' ')
    rect = [int(p) for p in rect]
    return rect

def check_image_size(img, hw, bbox):
    # check image size
    #print(hw, bbox)
    right_height, right_width = getHeightWidth(hw)
    #print(right_height, right_width, bbox)
    real_height, real_width, _ = img.shape
    
    if((bbox[0] > right_width) or (bbox[2] > right_width) or
         (bbox[1] >  right_height) or (bbox[3] > right_height)):
         return False, None
         
    elif (right_height != real_height) or (right_width != real_width):
        if abs(right_height/right_width - real_height/real_width) < 0.01:
            #print(abs(right_height/right_width - real_height/real_width))
            img = cv2.resize(img, (right_width, right_height))
            return True, img
        else:
            return False, None
    else:
        return True, img
    


def main(start_n, init_total_images, max_images, max_gallery, tag):

    with open('/storage/oprabhune/tianen_colab/face_recognition/IMDb/IMDb-Face.csv', mode ='r') as file:
        
        # reading the CSV file
        csvFile = csv.reader(file)
        id = ''
        gallery_counter = 0
        total_images = init_total_images
        #start_n = 0
        # displaying the contents of the CSV file
        for n, lines in enumerate(csvFile):
            if(n < start_n):
                #print('skipping')
                continue

            if(total_images > max_images):
                return n, total_images
            
            print(n, total_images)

            name, index, image, rect, hw, url = lines
            print(n, name, index, image, rect, hw, url)
            data = requests.get(url, stream = True)

            if data.ok:
                
                orig_img = Image.open(data.raw)
                if orig_img.mode == 'L':
                    # Convert the grayscale image to RGB
                    orig_img = orig_img.convert('RGB')
                
                orig_img = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR)  #BGR image

                check, orig_img = check_image_size(orig_img, hw, getCoordinate(rect))

                if(check):
                    print('check ok')
                    if(id != name):
                        id = name
                        gallery_counter = 0 

                    gallery_counter += 1
                    
                    #print('gallery_counter, max_gallery ', gallery_counter, max_gallery)
                    if(gallery_counter < max_gallery): #ensure no overlap between gallery and probe
                        #print('In inference, ', n)
                        # resize with pad
                        
                        orig_img, ratio, top, left = resize_with_pad(orig_img, (1080, 1080))
                        '''
                        # change coordinates of gt bbox
                        gt_bbox = getCoordinate(rect)
                        gt_bbox = [int(x*ratio) for x in gt_bbox] 
                        gt_bbox[0] += left
                        gt_bbox[2] += left
                        gt_bbox[1] += top
                        gt_bbox[3] += top
                        '''
                        ## 
                        
                        face_objs = DeepFace.extract_faces(orig_img,
                                                           target_size = (224, 224),
                                                           enforce_detection=False,
                                                           detector_backend = 'retinaface'
                                                           )
                        #orig_image_size = orig_img.shape
                        gt_mask = np.zeros_like(orig_img, dtype=np.uint8)
                        for face in face_objs:
                            if(face['confidence'] == 0):
                                continue
                            rect_start, rect_end = (face['facial_area']['x'], face['facial_area']['y']), (face['facial_area']['x'] + face['facial_area']['w'] ,face['facial_area']['y'] + face['facial_area']['h'] )
                            gt_mask = cv2.rectangle(gt_mask, rect_start, rect_end, (255, 255, 255), thickness=cv2.FILLED)
                        
                        '''
                        img_with_bbox = orig_img.copy()
                        for face in face_objs:
                            rect_start, rect_end = (face['facial_area']['x'], face['facial_area']['y']), (face['facial_area']['x'] + face['facial_area']['w'] ,face['facial_area']['y'] + face['facial_area']['h'] )
                            img_with_bbox = cv2.rectangle(img_with_bbox, rect_start, rect_end, (255, 255, 255), thickness=2)
                        '''

                        total_images += 1
                        cv2.imwrite(f"face_roi_dataset/{tag}_masks/{name}_{image}", gt_mask.astype(np.uint8))
                        cv2.imwrite(f"face_roi_dataset/{tag}_images/{name}_{image}", orig_img)
                        

if __name__ == "__main__":
    n, total_images = main(2618, 573, 5000,  10, tag = 'train')
    
    main(n, 0, 2000,  10, tag = 'test')
    
