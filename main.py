import glob

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
from skimage import io
from skimage.color import gray2rgb, rgb2gray
from skimage.feature import canny


def adaptive_threshold(path,block_size,threshold_type = cv2.THRESH_BINARY):
    '''This function implements adaptive thresholding to generate masks for pre-training.
     threshold value is calculated for smaller regions.
     This leads to different threshold values for different regions with respect to the change in lighting.

     path : Path to the folder containing the images.
     threshold_type (optional): The type of thresholding to be done in smaller regions of the image. Ex: cv.THRESH_BINARY.
     block_size : The blockSize determines the size of the neighbourhood area
     and C is a constant that is subtracted from the mean or weighted sum of the neighbourhood pixels.
     '''

    image = cv2.imread(path) # -1 argument is used to make OpenCV read 16-bit images.
    
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # change the image to grayscale to make it compatible for thresholding.

    mean = np.mean(gray_image)

    if mean <=4.0 :

        gray_image[gray_image>1.5*mean] *= 155
        max = np.max(gray_image)
        cv2.imwrite('gray_image_min.tif', gray_image)
        thresh_image = cv2.adaptiveThreshold(gray_image,max,cv2.ADAPTIVE_THRESH_MEAN_C,threshold_type,block_size,-mean) #Adaptive thresholding

    else:
        max = np.max(gray_image)
        thresh_image = cv2.adaptiveThreshold(gray_image, max, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size,-2 * mean) #Adaptive thresholding
    



def zoom(img, zoom_factor=1.5):
    #Zoom by a factor of 1.5
    return cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor,interpolation= cv2.INTER_LINEAR)

def zoom_generator():
    '''This function takes as 16-bit raw images and generates their corresponding zoomed images.'''

    path = "D:/DeDustProject/Artifacts/Artifacts/dapi_6_exp_times/*.tif"
    filename = ''
    for file in glob.glob(path):

        filename = str(file)
        raw_image = cv2.imread(file,-1)  # -1 argument is used to make OpenCV read 16-bit images.
        cropping_factor = 6
        cropping_height, cropping_width = raw_image.shape[0]//cropping_factor,raw_image.shape[1]//cropping_factor
        height_offset = 0 # Used to vertically traverse a single image.

        while cropping_height + height_offset <= raw_image.shape[0]:
            width_offset = 0

            while cropping_width + width_offset <= raw_image.shape[1]:
                cropped_image = raw_image[height_offset:cropping_height + height_offset,width_offset:cropping_width+width_offset]
                # width_offset is used to horizontally traverse a single image.
                zoomed_and_cropped_image = zoom(cropped_image)

                width_offset += 108 # raw_image.shape[0]/20 = 108. This gives 20 zoomed images.
                yield zoomed_and_cropped_image,filename

            height_offset += cropping_height # Move to next row.


def plot_from_logs(path_to_log_file):
    df = pd.DataFrame()
    df = pd.read_csv(path_to_log_file,delim_whitespace=True)
    cols = df.columns
    figure, ax1 = plt.subplots(1,3)
    figure.subplots_adjust(hspace=0.5, wspace=0.5)
    figure.set_size_inches(18.5, 10.5)
    #print(ax1.shape)
    ax1[0].plot(df[cols[0]], df[cols[2]], linewidth=0.5, zorder=1, label="Force1")
    ax1[0].set_xlabel('epochs')
    ax1[0].set_ylabel('l_d2')
    ax1[1].plot(df[cols[0]], df[cols[3]], linewidth=0.5, zorder=1, label="Force2")
    ax1[1].set_xlabel('epochs')
    ax1[1].set_ylabel('l_g2')
    ax1[2].plot(df[cols[0]], df[cols[4]], linewidth=0.5, zorder=1, label="Force2")
    ax1[2].set_xlabel('epochs')
    ax1[2].set_ylabel('l_l1')
    plt.title('edge_connect_all_losses_high_LR_10_times')
    plt.show()




def canny_detector(path_to_image):
    img = io.imread(path_to_image)
    gray_img = rgb2gray(img) * 255
    gray_img = np.asarray(gray_img,dtype='uint8')
    mask  = io.imread("D:/DeDustProject/data/final_images_zoomed/ir_masks_edge/irregularmask_103.PNG")
    edges = canny(gray_img,sigma=1, low_threshold=20, high_threshold=40,mask=None)
    plt.subplot(121), plt.imshow(gray_img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()




