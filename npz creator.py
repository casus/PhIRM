import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt


pathto_files = 'D:/DeDustProject/data/zoomed_images/*.tif'

data = []
count = 0
for file in glob.glob(pathto_files):
      single_im = cv2.imread(file,-1)
      single_array = np.array(single_im,dtype='uint16')
      data.append(single_array)

#np.savez('zoomed_imgs.npz',data=data)
data = np.load('../data/zoomed_imgs.npz')
print(data)
train_data = data['data']
rearranged_arr = np.moveaxis(train_data, [1, 0], [1, 2])
print(rearranged_arr.shape)
input_img = rearranged_arr[...,0]
nucleus_img = rearranged_arr[..., 2]
print(input_img.shape)


cv2.imshow('test1',input_img[100:300,200:400])
cv2.waitKey(0)

fig, ax = plt.subplots(1,2)
ax[0].imshow(input_img[100:300,200:400],cmap='gray')
ax[0].axis('off')
ax[1].imshow(nucleus_img[100:300,200:400],cmap='gray')
ax[1].axis('off')