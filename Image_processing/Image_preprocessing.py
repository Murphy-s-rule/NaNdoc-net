import pandas as pd

csv = pd.read_csv(r'D:\Data\snukb_dataset\train/train.csv')
csv

#%%
import os
import cv2
import cv2 as cv
import re
import numpy as np

from matplotlib import pyplot as plt

img_path = r'D:\Data\OCR\train\images'
out_path = r'D:\Data\OCR\Image_Processing'
img_list = os.listdir(img_path)
img_list.sort(key=lambda f: int(re.sub('\D', '', f)))


# =============================================================================
# https://medipixel.io/products/blog/Tutorial-Basic-Image-Operation
# https://medipixel.io/products/blog/Tutorial-Histogram
# =============================================================================

plt.ioff()
for i, img_name in enumerate(img_list):
    img = cv2.imread(f'{img_path}/{img_name}', cv2.IMREAD_COLOR)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # image dimension matching
    
    #%% Gaussian noise reduction
    # 쉼표, 콜론 등을 없애지 않도록 해야함
    img2 = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
    # plt.subplot(121),plt.imshow(img)
    # plt.subplot(122),plt.imshow(img2)
    # plt.show()
    
    #%% image hole filling
    # gaussian noise reduction 후에 시행하면 좋다
    # kernel = np.ones((3, 3), np.uint8)
    # erosion = cv2.erode(img2, kernel, iterations=1)
    # dilation = cv2.dilate(img2, kernel, iterations=1)
    # cv2.imshow("erosion", erosion)
    # cv2.imshow("dilation", dilation)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
    
    #%% image binarize / thresholding
    # ret, dst = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    # 임의의 threshold는 매우 성능이 안좋다.
    
    # 통계적 binarize. Otsu thresholding
    # gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # t, dst = cv2.threshold(gray2, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # # blurr한 이미지와 배경 color에 취약하다.
    # # plt.imshow(dst)
    # plt.subplot(121);plt.imshow(img)
    # plt.subplot(122);plt.imshow(dst, cmap='gray')
    # plt.savefig(f'{out_path}/{img_name}.jpg')
    
    #%% image normalization
    # img_norm2 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    # plt.subplot(121);plt.imshow(img)
    # plt.subplot(122);plt.imshow(dst, cmap='gray')
    # plt.savefig(f'{out_path}/{img_name}.jpg')
    
    #%% image histogram normalization
    # img2 = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR) 
    
    #%% image histogram equalization
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV) 
    
    #YUV 컬러 스케일의 첫번째 채널에 대해서 이퀄라이즈 적용
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0]) 
    
    #컬러 스케일을 YUV에서 BGR로 변경
    img3 = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR) 
    
    #  noise reduction
    # img4 = cv2.fastNlMeansDenoisingColored(img3,None,10,10,7,21)
    
    plt.subplot(121);plt.imshow(img)
    plt.subplot(122);plt.imshow(img4, cmap='gray')
    plt.savefig(f'{out_path}/{img_name}.jpg')
    
    
    #%% image sharpening
    kernel_sharpen_1 = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]]) 
    #정규화를 하지 않은 이유는 모든 값을 다 더하면 1이되기때문에 1로 나눈것과 같은 효과
    kernel_sharpen_2 = np.array([[1,1,1],[1,-7,1],[1,1,1]]) #이것도 마찬가지
    kernel_sharpen_3 = np.array([[-1,-1,-1,-1,-1],[-1,2,2,2,-1],[-1,2,8,2,-1],[-1,2,2,2,-1],[-1,-1,-1,-1,-1]])/8.0 
    #정규화위해 8로나눔
    
    #applying different kernels to the input image
    output_1 = cv2.filter2D(img,-1,kernel_sharpen_1)
    output_2 = cv2.filter2D(img,-1,kernel_sharpen_2)
    output_3 = cv2.filter2D(img,-1,kernel_sharpen_3)
    
    cv2.imshow('Origin',img)
    cv2.imshow('Sharpening',output_1)
    cv2.imshow('Excessive Sharpening',output_2)
    cv2.imshow('Edge Enhancement',output_3)
    cv2.waitKey()

    
    #%% edge detection
    
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, 3)
    laplacian = cv2.Laplacian(gray, cv2.CV_8U, ksize=3)
    canny = cv2.Canny(img, 100, 255)
    
    cv2.imshow("sobel", sobel)
    cv2.imshow("laplacian", laplacian)
    cv2.imshow("canny", canny)
    cv2.waitKey()
    cv2.destroyAllWindows()
    
    # %%blurr removal

    
    # iamge skew correction
    1
