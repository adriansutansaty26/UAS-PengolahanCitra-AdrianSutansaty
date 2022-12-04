import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('sample.png');
img_gray = cv.imread('sample.png',0) # 1. KONVERSI GAMBAR SAMPLE MENJADI GRAYSCALE 

t = 128 # 2. INISIASI NILAI TENGAH DARI GRAYSCALE 
H,W = img_gray.shape[:2]
img_binary = img_gray.copy()

# 3. OPERASI KONVERSI PIKSEL KE BINER
for i in range(H):
    for j in range(W):
        if img_binary[i,j] >= t: # PIKSEL LEBIH DARI ATAU SAMA DENGAN 128
            img_binary[i,j] = 255
        elif img_binary[i, j] < t: # PIKSEL KURANG DARI 128
            img_binary[i,j] = 0
            



plt.subplot(2,2,1),plt.imshow(img_binary,cmap = 'gray')
plt.title('Result'), plt.xticks([]), plt.yticks([])
plt.show()            
            
cv.imwrite('Result.png', img_binary)
cv.waitKey(0)
cv.destroyAllWindows()