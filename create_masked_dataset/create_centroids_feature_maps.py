from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import random
import os
import cv2
import shutil
### For visualizing the outputs ###
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

phase = 'val'

annFile = f'/Users/omkarp/Downloads/instances_{phase}2017.json'
coco=COCO(annFile)
# Load the categories in a variable
catIDs = coco.getCatIds()
cats = coco.loadCats(catIDs)

print(cats)

# Define the classes (out of the 81) which you want to see. Others will not be shown.
filterClasses = ['person']

# Fetch class IDs only corresponding to the filterClasses
catIds = coco.getCatIds(catNms=filterClasses) 
# Get all images containing the above Category IDs
imgIds = coco.getImgIds(catIds=catIds)
print("Number of images containing all the  classes:", len(imgIds))

for number, imgId in enumerate(imgIds):
  if(number % 1000 == 0):
    print(number)
  centroids= []
  img = coco.loadImgs(imgId)[0]

  src = f"/Users/omkarp/Downloads/{phase}2017/{img['file_name']}"
  dstn = f"/Users/omkarp/Documents/tianen_project_datasets/centroids_approach_dataset/{phase}_images/{img['file_name']}"
  shutil.copyfile(src, dstn)

  annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
  #print(annIds)
  anns = coco.loadAnns(annIds)
  for ann in anns:
    #print(ann)
    centroid = (int(ann['bbox'][0] + (ann['bbox'][2]/2.0)), int(ann['bbox'][1] + (ann['bbox'][3]/2.0)))
    centroids.append(centroid)
  mask = np.zeros((16, 16))
  #print(centroids)
  for centroid in centroids:
    centroid_in_500x500 = (centroid[1] * 500 / img['height'], centroid[0] * 500 / img['width'])
    centroid_in_16x16 = (int(centroid_in_500x500[1] * 16 / 500), int(centroid_in_500x500[0] * 16 / 500 ))
    '''
    print(centroid)
    print(centroid_in_500x500)
    print(centroid_in_16x16)
    '''
    mask[centroid_in_16x16[1], centroid_in_16x16[0]] = 255

  mask_file = f"/Users/omkarp/Documents/tianen_project_datasets/centroids_approach_dataset/{phase}_masks/{img['file_name']}"
  cv2.imwrite(mask_file, mask)
  #print('\n')