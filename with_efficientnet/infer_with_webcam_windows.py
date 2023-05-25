from EfficientNetWithConv import EfficientNetWithConv
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
from PIL import Image
from matplotlib import cm
# load the model, data transforms
model = EfficientNetWithConv().to(DEVICE)
model.load_state_dict(torch.load('my_checkpoint.pth.tar', map_location=DEVICE)["state_dict"])

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )


def infer_from_file(img_path):
    image = np.array(Image.open(img_path).convert("RGB"))
    print('Shape : ', image.shape)
    input_image = val_transforms(image=image)['image']

    out = torch.sigmoid(model(input_image.unsqueeze(0)))
    binary_mask = (out > 0.35).float() 
    colored_feature_map = cm.viridis(out.squeeze().detach().numpy())
    cv2.imwrite('colored_feature_map.png', colored_feature_map* 255)
    out = (out.squeeze() * 255).detach().numpy()
    binary_mask = (binary_mask.squeeze() * 255).detach().numpy()
    cv2.imwrite('feature_map.png', out)
    cv2.imwrite('binary_mask.png', binary_mask)

    return 
    '''
    # Display the resulting frame
    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    cv2.namedWindow("feature map", cv2.WINDOW_NORMAL)
    cv2.namedWindow("thresholded feature map", cv2.WINDOW_NORMAL)

    cv2.resizeWindow("frame", 512, 512)
    cv2.resizeWindow("feature map", 512, 512)
    cv2.resizeWindow("thresholded feature map", 512, 512)

    
    cv2.imshow('frame', image)
    cv2.imshow('feature map', out)
    cv2.imshow('thresholded feature map', binary_mask)
    cv2.waitKey(0)
    '''

def infer_from_webcam():
    # define a video capture object
    vid = cv2.VideoCapture(0)

    vid.set(cv2.CAP_PROP_FRAME_WIDTH, 512) # set the width to 640 pixels
    vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 512) # set the height to 480 pixels

    while(True):
        
        # Capture the video frame
        # by frame
        ret, image_ = vid.read()
        #print(image.shape)
        
        # process the frames
        
        image = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)
        #print(image.shape)
        tensor_img_A = val_transforms(image=np.array(image))['image']
        #out = (torch.sigmoid(model(tensor_img_A.unsqueeze(0))).squeeze(0, 1) * 255).detach().numpy()

        out = torch.sigmoid(model(tensor_img_A.unsqueeze(0)))
        binary_mask = (out > 0.35).float() 

        out = (out.squeeze(0, 1) * 255).detach().numpy()
        binary_mask = (binary_mask.squeeze(0, 1) * 255).detach().numpy()
        


        # Display the resulting frame
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.namedWindow("feature map", cv2.WINDOW_NORMAL)
        cv2.namedWindow("thresholded feature map", cv2.WINDOW_NORMAL)

        cv2.resizeWindow("frame", 512, 512)
        cv2.resizeWindow("feature map", 512, 512)
        cv2.resizeWindow("thresholded feature map", 512, 512)

        
        cv2.imshow('frame', image_)
        cv2.imshow('feature map', out)
        cv2.imshow('thresholded feature map', binary_mask)
        
        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    infer_from_file('test_image.jpeg')
    #infer_from_webcam()