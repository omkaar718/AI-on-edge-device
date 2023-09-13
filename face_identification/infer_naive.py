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

def write_image(file_name, data):
    f = open(file_name,'wb')
    f.write(data)
    f.close()

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

def faceCrop(img, xmin, ymin, xmax, ymax, scale_ratio):
    '''
    crop face from image, the scale_ratio used to control margin size around face.
    using a margin, when aligning faces you will not lose information of face
    '''
    if type(img) == str:
        img = cv2.imread(img)

    hmax, wmax, _ = img.shape
    x = (xmin + xmax) / 2
    y = (ymin + ymax) / 2
    w = (xmax - xmin) * scale_ratio
    h = (ymax - ymin) * scale_ratio
    # new xmin, ymin, xmax and ymax
    xmin = x - w/2
    xmax = x + w/2
    ymin = y - h/2
    ymax = y + h/2

    xmin = max(0, int(xmin))
    ymin = max(0, int(ymin))
    xmax = min(wmax, int(xmax))
    ymax = min(hmax, int(ymax))
    
    face = img[ymin:ymax,xmin:xmax,:]
    return face


def save_crop(img, image_file, rect, scale_ratio):
    # crop face from image
    location = getCoordinate(rect)
    face = faceCrop(img, *location, scale_ratio)
    cv2.imwrite(image_file, face)

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
    
def find_IoU(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou

def main(start_n, end_n, max_identities, max_gallery, gallery_folder):
    #new_model = tf.keras.models.load_model('smaller_model_saved_during_training_alpha_0.1_240_15')
    #print('Model loaded')
    #rf_range = np.arange(0, 1.05, 0.05)
    #sf_range = np.arange(0.05, 1.05, 0.05)
    #rows, cols = (len(rf_range), len(sf_range))
    resolutions = [16, 32, 64, 128, 200, 400, 600, 800, 1080]
    results = np.zeros(len(resolutions))
    results_in_fraction = np.zeros(len(resolutions))
    save_flag = False
    with open('/storage/oprabhune/tianen_colab/face_recognition/IMDb/IMDb-Face.csv', mode ='r') as file:
        
        # reading the CSV file
        csvFile = csv.reader(file)
        id = ''
        gallery_counter = 0
        probe_counter = 0
        n_identities = 0
        #start_n = 0
        # displaying the contents of the CSV file
        for n, lines in enumerate(csvFile):
            if(n == 0):
                continue
            

            if(n < start_n):
                continue

            if(n > end_n):
                print('n', n-1)
                print('probe counter : ', probe_counter)
                print('results_in_fraction : \n',results_in_fraction)
                print('results \n', results)
                break

            name, index, image, rect, hw, url = lines
            print(n, name, index, image, rect, hw, url)
            data = requests.get(url, stream = True)

            if data.ok:
                
                #print('\n\n',image)
                #data = data.content
                orig_img = Image.open(data.raw)
                if orig_img.mode == 'L':
                    # Convert the grayscale image to RGB
                    orig_img = orig_img.convert('RGB')
                
                orig_img = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR)  #BGR image

                check, orig_img = check_image_size(orig_img, hw, getCoordinate(rect))

                if(check):
                    

                    if(id != name):
                        n_identities += 1
                        if(n_identities > max_identities):
                            print('\nMax identities reached!!')
                            print(n, probe_counter)
                            break
                        
                        if(n_identities % 50 == 0):
                            print('n_identities: ', n_identities, name)
                        #print(id, name, image)
                        id = name
                        gallery_counter = 0 

                    gallery_counter += 1
                    
                    if(gallery_counter > max_gallery):
                        probe_counter += 1
                        
                        for resolution_i, resolution in enumerate(resolutions):
                            
                            img_2, ratio, top, left = resize_with_pad(orig_img, (resolution, resolution))
                            #cv2.imwrite('resized.jpg', img_2)

                            # change coordinates of gt bbox
                            gt_bbox = getCoordinate(rect)
                            gt_bbox = [int(x*ratio) for x in gt_bbox] 
                            gt_bbox[0] += left
                            gt_bbox[2] += left
                            gt_bbox[1] += top
                            gt_bbox[3] += top
                            
                            # DO FACE RECOG HERE
                            #img_2 = orig_img

                            dfs = DeepFace.find(img_2, 
                                            db_path = gallery_folder, 
                                            enforce_detection = False,
                                            detector_backend = 'retinaface',
                                            model_name = "ArcFace"
                                    )
                            for df_i, df in enumerate(dfs):
                                        
                                        #if(df.empty):
                                            #print('No match found')
                                        if(not df.empty):
                                            #print(df)
                                            prediction = df.iloc[0]['identity'].split('/')[1]
                                            #print(df.iloc[0]['identity'])
                                            #print('Match found with ', prediction)
                                            if(prediction == name):
                                                pred_bbox = (df.iloc[0]['source_x'],
                                                            df.iloc[0]['source_y'],
                                                            df.iloc[0]['source_x'] + df.iloc[0]['source_w'],
                                                            df.iloc[0]['source_y'] + df.iloc[0]['source_h']
                                                )

                                                #gt_bbox = getCoordinate(rect)
                                                
                                                '''
                                                color = (255, 0,0)
                                                color_2 = (0, 255, 255)
                                                thickness = 1
                                                img_with_bbox = cv2.rectangle(img_2, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), color, thickness)
                                                img_with_bbox = cv2.rectangle(img_2, (pred_bbox[0], pred_bbox[1]), (pred_bbox[2], pred_bbox[3]), color_2, thickness)
                                                
                                                print('\nSaving image at resolution ', resolution)
                                                cv2.imwrite('img_with_bbox.jpg', img_with_bbox)
                                                '''


                                                if(find_IoU(pred_bbox, gt_bbox) > 0.5):
                                                    #print('Good IoU')
                                                    
                                                    #print(rf_i, sf_i)
                                                    #print('Modifying results - current  = \n', results)
                                                    results[resolution_i] +=1
                                                    #print('After modify = \n', results)
                                                else:
                                                    pass
                                                    #print('IoU NOT GOOD')
                                                   
                                                            

                            
                        results_in_fraction = results/probe_counter

                        print('n : ', n)
                        print('probe counter : ', probe_counter)
                        print('results_in_fraction : \n',results_in_fraction)
                        with open(f'naive_results_max-identities_{max_identities}_in_fraction.npy', 'wb') as f:
                            np.save(f, results_in_fraction)

                        with open(f'naive_results_max-identities_{max_identities}.npy', 'wb') as f:
                            np.save(f, results)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_n', type=int, default=1)
    parser.add_argument('--end_n', type=int, default=1000000)

    parser.add_argument('--create_gallery', action='store_true')
    parser.add_argument('--max_identities', type=int, default=25)
    parser.add_argument('--max_gallery', type=int, default=10)
    parser.add_argument('--gallery_folder', type=str, default='/storage/oprabhune/tianen_colab/face_recognition/IMDb/gallery_100')
    args = parser.parse_args()
    
    main(args.start_n, args.end_n, args.max_identities, args.max_gallery, args.gallery_folder)
