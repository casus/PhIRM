import math
import os

import cv2
#from data_augmentation import min_max_255
import numpy as np
from matplotlib import pyplot as plt
from skimage import img_as_ubyte, io
from skimage import transform as t
from skimage import util
from skimage.filters import threshold_otsu


def iou_coef(y_true, y_pred, smooth=1):
    '''
    This function calculates the IoU coeffient between two images.
    '''
    y_pred = np.asarray(y_pred, dtype='float32')
    y_pred /= 255
    y_true = np.asarray(y_true, dtype='float32')
    y_true /= 255
    intersection = np.sum(np.abs(y_true * y_pred), axis=0) + np.sum(np.abs(y_true * y_pred), axis=1)
    union = np.sum(y_true,axis=0) + np.sum(y_true,axis=1) + np.sum(y_pred,axis=0) + np.sum(y_pred,axis=1)  - intersection
    iou = np.mean((intersection + smooth ) / (union + smooth))
    return iou


def dice_coef(y_true, y_pred):
    '''
    This function calculates the dice coeffient between two images.
    '''
    
    y_pred = np.asarray(y_pred,dtype='float32')
    y_pred /= 255
    y_true = np.asarray(y_true, dtype='float32')
    y_true /= 255    
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true + y_pred)) #(2 * Area of Overlap)/(total pixels combined)


def inpaintMetricScorer(diff_in_nuclei_number,diff_in_nuclei_area,diff_in_artifact_number,diff_in_artifact_area):
    '''
    This fucntion takes as inout the differences in a source and inpainted image and calculates the final inpainting score.
    '''
    nuclei_number_loss_weight = 1.1 #best is 1.1
    nuclei_area_loss_weight = 0.0002 #best is 0.0002
    artifact_area_loss_weight = 0.001 #best is 0.001
    
    

    if diff_in_nuclei_number == 0:
        nuclei_number_loss = 0.0

    elif diff_in_nuclei_number > 0:
        nuclei_number_loss = (nuclei_number_loss_weight ** diff_in_nuclei_number) * diff_in_nuclei_number

    else:
        nuclei_number_loss = 2 ** abs(diff_in_nuclei_number)

    nuclei_area_loss = nuclei_area_loss_weight * diff_in_nuclei_area

    

    artifact_area_loss = artifact_area_loss_weight * diff_in_artifact_area

    

    total_loss = nuclei_number_loss + nuclei_area_loss  + artifact_area_loss

    # print(f'[INFO] Nuclei number loss : ',nuclei_number_loss,'Nuclei area loss : ',nuclei_area_loss
    #       ,'Artifact area loss : ',artifact_area_loss , 'Image naturalness loss :',image_naturalness_loss,'Total loss is : ',total_loss)

    inpaintScore = 10 - total_loss

    if inpaintScore < 0:
        inpaintScore = 0.0

    #print(f'Inpaint score is :', inpaintScore)
    return inpaintScore




def inpaintMetric(path_to_source_images,path_to_inpainted_images):
    '''
    This fucntion calculates the differences between the source and inpainted images using the factors present in the inpainting metric.
    '''


    if os.path.isdir(path_to_source_images) and os.path.isdir(path_to_inpainted_images):

        mode = 2
        source_images_filelist = os.listdir(path_to_source_images)
        inpainted_images_filelist = os.listdir(path_to_inpainted_images)

        number_of_source_images = len(source_images_filelist)
        number_of_inpainted_images = len(inpainted_images_filelist)
        score_array = np.zeros((number_of_source_images),dtype='float32')
        if number_of_source_images != number_of_inpainted_images:

            print('Number of source and inpainted images do not match!')
            print('Exiting')
            return

        else:

            for source_file, inpaint_file in zip(source_images_filelist,inpainted_images_filelist):

                path_to_source_image = os.path.join(path_to_source_images,source_file)
                number_of_source_nuclei,total_area_of_source_nuclei,number_of_artifacts_in_source,total_area_of_source_artifacts = stats_calculator(path_to_source_image)
                path_to_inpainted_image = os.path.join(path_to_inpainted_images, inpaint_file)
                number_of_inpaint_nuclei, total_area_of_inpaint_nuclei, number_of_artifacts_in_inpaint, total_area_of_inpaint_artifacts = stats_calculator(path_to_inpainted_image)

                diff_in_nuclei_number = number_of_source_nuclei - number_of_inpaint_nuclei
                diff_in_nuclei_area = abs (total_area_of_source_nuclei - total_area_of_inpaint_nuclei)
                diff_in_artifact_number = abs (number_of_artifacts_in_inpaint - number_of_artifacts_in_source)
                diff_in_artifact_area = abs(total_area_of_inpaint_artifacts - total_area_of_source_artifacts)
                

                inpaintScore = inpaintMetricScorer(diff_in_nuclei_number,diff_in_nuclei_area,diff_in_artifact_number,diff_in_artifact_area)
                score_array.append(inpaintScore)

    elif os.path.isfile(path_to_source_images) and os.path.isfile(path_to_inpainted_images):
        
        path_to_source_image = path_to_source_images
        number_of_source_nuclei, total_area_of_source_nuclei, number_of_artifacts_in_source, total_area_of_source_artifacts = stats_calculator(
                path_to_source_image)


        path_to_inpainted_image = path_to_inpainted_images
        number_of_inpaint_nuclei, total_area_of_inpaint_nuclei, number_of_artifacts_in_inpaint, total_area_of_inpaint_artifacts = stats_calculator(
                path_to_inpainted_image)

        diff_in_nuclei_number = number_of_source_nuclei - number_of_inpaint_nuclei
        diff_in_nuclei_area = abs(total_area_of_source_nuclei - total_area_of_inpaint_nuclei)
        diff_in_artifact_number = abs(number_of_artifacts_in_inpaint - number_of_artifacts_in_source)
        diff_in_artifact_area = abs(total_area_of_inpaint_artifacts - total_area_of_source_artifacts)
       
        
        inpaintScore = inpaintMetricScorer(diff_in_nuclei_number, diff_in_nuclei_area, diff_in_artifact_number,
                                           diff_in_artifact_area)



def stats_calculator(path_to_source_image):     

    source_image = cv2.imread(path_to_source_image)
              
    io_image = io.imread(path_to_source_image)
    imgray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    threshold = threshold_otsu(imgray)  # Otsu thresholding          
    binary = imgray > threshold * 1.1  # Masked image.Multiplied by 0.7 to produce better masks.
    binary = img_as_ubyte(binary)

    # apply connected component analysis to the thresholded image   
    output = cv2.connectedComponentsWithStats(
        binary, 8, cv2.CV_32S)                                            

    (numLabels, labels, stats, centroids) = output
    mask = np.zeros(imgray.shape, dtype="uint8")
    number_of_nuclei = 0
    total_nuclei_area = 0
    number_of_artifacts = 0
    total_artifacts_area = 0
    rect = np.zeros(imgray.shape, dtype="uint8")
    # loop over the number of unique connected component labels, skipping
    # over the first label (as label zero is the background)
    
    for i in range(1, numLabels):                                 
        # extract the connected component statistics for the current
        # label
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        keepArea = area > 50

        if keepArea:

            # number_of_components += 1
            # total_components_area += area

            rect = io_image[y:y + h, x:x + w]
            mean = np.average(rect)                             
            max = np.max(rect)

            if max == 255:
                rect[rect < 30] = 255
                mean = rect.mean()

                if mean >= 210:
                    number_of_artifacts += 1
                    total_artifacts_area += area

                else:
                    number_of_nuclei += 1
                    total_nuclei_area += area
            else:
                if area < 2200:
                     number_of_nuclei += 1
                else:
                    number_of_nuclei += 1
                total_nuclei_area += area

    return (number_of_nuclei, total_nuclei_area, number_of_artifacts, total_artifacts_area)

