import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def detectEdge(img):
    
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    rows = len(img)
    cols = len(img[0])
    
    # 1. OPERASI SOBEL DERIVATIVE X ( MENDETEKSI GARIS TEPI HORIZONTAL )
    Gx = np.array(np.mat('1 0 -1; 2 0 -2; 1 0 -1'))
    sobelXimg = np.zeros((rows,cols))
    for i in range(0,rows-3):
        for j in range(0,cols-3):
            
            image = np.zeros((3,3))
            a = i
            b = j
            for k in range(0,3):
                b = j
                for l in range(0,3):
                    image[k,l] = img[a,b]
                    b = b+1
                a = a +1
             
            row1total = image[0,1]*Gx[0,1] + image[0,2]*Gx[0,2] + image[0,0]*Gx[0,0]
            row2total = 0
            row3total = image[2,1]*Gx[2,1] + image[2,2]*Gx[2,2] + image[2,0]*Gx[2,0]
            rowtotal = row1total + row2total + row3total
            sobelXimg[i,j] = rowtotal
    
    # 2. OPERASI SOBEL DERIVATIVE Y ( MENDETEKSI GARIS TEPI VERTIKAL )
    Gy = np.array(np.mat('1 2 1; 0 0 0; -1 -2 -1'))
    sobelYimg = np.zeros((rows,cols))
    for i in range(0,rows-3):
         for j in range(0,cols-3):
             
             image = np.zeros((3,3))
             a = i
             b = j
             for k in range(0,3):
                 b = j
                 for l in range(0,3):
                     image[k,l] = img[a,b]
                     b = b+1
                 a = a +1
                
             row1total = image[0,1]*Gy[0,1] + image[0,2]*Gy[0,2] + image[0,0]*Gy[0,0]
             row2total = 0
             row3total = image[2,1]*Gy[2,1] + image[2,2]*Gy[2,2] + image[2,0]*Gy[2,0]
             rowtotal = row1total + row2total + row3total
             sobelYimg[i,j] = rowtotal
             
    # 3. OPERASI MAGNITUDE = Nilai Absolut SOBEL X + Nilai Absolut SOBEL Y
    magnitudeImg = np.zeros([rows, cols])
    for i in range(0,rows):
        for j in range(0,cols):
            magnitudeImg[i,j] = abs(sobelXimg[i, j]) + abs(sobelYimg[i, j])
            
    return sobelXimg, sobelYimg, magnitudeImg


def edgeFill(img, imgEdge):
    
    # Mengkonversi pixel ke grayscale & membaca ukuran piksel
    imgEdge = np.uint8(imgEdge)
    hh, ww = img.shape[:2]
    
    # Melakukan operasi thresholding untuk meningkatkan kejelasan dari tepi
    thresh = cv.threshold(imgEdge, 30, 255, cv.THRESH_BINARY)[1]
    
    # Mendapatkan garis tepi dengan kontur yang terluas
    contours = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv.contourArea)
    
    # Mengubah isi piksel kontur terluas menjadi putih untuk dijadikan mask
    mask = np.zeros_like(imgEdge)
    cv.drawContours(mask, [big_contour], 0, (255,255,255), cv.FILLED)
    
    # Melakukan operasi bitwise untuk menggabungkan antara mask dengan gambar asli
    result = cv.bitwise_and(img,img, mask= mask)
    return mask, result





img = cv.imread('sample.jpg');
imgEdge = detectEdge(img)
result = edgeFill(img, imgEdge[2])

plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('1. Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(imgEdge[0],cmap = 'gray')
plt.title('2. Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(imgEdge[1],cmap = 'gray')
plt.title('3. Sobel Y'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(imgEdge[2],cmap = 'gray')
plt.title('4. Magnitude'), plt.xticks([]), plt.yticks([])
plt.show()

plt.subplot(2,2,1),plt.imshow(result[0],cmap = 'gray')
plt.title('5. Mask'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(result[1],cmap = 'gray')
plt.title('Result : Edge Detection'), plt.xticks([]), plt.yticks([])
plt.show()

cv.imwrite('Result.jpg', result[1])

cv.waitKey(0)
cv.destroyAllWindows()