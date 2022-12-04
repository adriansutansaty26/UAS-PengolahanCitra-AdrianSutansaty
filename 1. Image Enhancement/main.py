# Library
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


def imgEnhancement(img, img_mask):
    
    # 1. Operasi bitwise untuk menggabungkan antara gambar original dengan mask (Output ke-1)
    result_bitwise = cv.bitwise_and(img,img, mask= img_mask)

    # 2. Noise Reduction dengan median filter 3x3 (Output ke-2)
    img_noisy = result_bitwise
    m, n = img_noisy.shape
    res_noise_removal = np.zeros([m, n])
    for i in range(1, m-1):
        for j in range(1, n-1):
            temp = [img_noisy[i-1, j-1],
                   img_noisy[i-1, j],
                   img_noisy[i-1, j + 1],
                   img_noisy[i, j-1],
                   img_noisy[i, j],
                   img_noisy[i, j + 1],
                   img_noisy[i + 1, j-1],
                   img_noisy[i + 1, j],
                   img_noisy[i + 1, j + 1]]
            temp = sorted(temp)
            res_noise_removal[i, j]= temp[4]
    result_noise_removal = res_noise_removal.astype(np.uint8)

    # 3. Perbaikan kontras dengan histogram equalization (Output ke-3)
    result_histogram_eq = cv.equalizeHist(res_noise_removal.astype('uint8'))

    # Final. Melakukan thresholding (Output Result)
    ret, result = cv.threshold(result_histogram_eq, 120, 255, cv.THRESH_TOZERO)
    
    return result_bitwise, result_noise_removal, result_histogram_eq, result


img = cv.imread('original.png', 0)
img_mask = cv.imread('mask.png', 0)

result = imgEnhancement(img, img_mask)

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(img_mask,cmap = 'gray')
plt.title('Mask'), plt.xticks([]), plt.yticks([])
plt.show()

plt.subplot(2,2,1),plt.imshow(result[0],cmap = 'gray')
plt.title('1. Bitwise'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(result[1],cmap = 'gray')
plt.title('2. Noise Removed'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(result[2],cmap = 'gray')
plt.title('3. Histogram Equalized'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(result[3],cmap = 'gray')
plt.title('4. Result'), plt.xticks([]), plt.yticks([])
plt.show()

cv.imwrite('1. Bitwise.jpg', result[0])
cv.imwrite('2. Noise Removed.jpg', result[1])
cv.imwrite('3. Histogram Equalized.jpg', result[2])
cv.imwrite('4. Result.jpg', result[3])

cv.waitKey(0)
cv.destroyAllWindows()