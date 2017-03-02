#encoding=utf-8
import mahotas as mh
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

image = mh.imread('2.jpg')


def gini_coef(wealths):
    N = 10
    wealths = sorted(wealths)
    sum_wealths = sum(wealths)
    l = len(wealths)
    step = l/N
    steparr = []
    for i in range(N):
        steparr.append(i*step)
    steparr.append(l-1)
    w_num = []
    tmpsum = 0
    for i in range(N-1):
        for j in range(steparr[i], steparr[i+1]):
            tmpsum += wealths[j]
        w_num.append(float(tmpsum)/sum_wealths)

    ans = sum(w_num)
    return  1.0 - (2.0*ans + 1.0)/(N+1)


#二值化
image = mh.colors.rgb2gray(image, dtype=np.uint8)
plt.gray()
thresh = mh.thresholding.otsu(image)
binarized = (image > thresh)
#plt.imshow(binarized)
#plt.show()

dataset = []

a, b = binarized.shape
for i in range(a):
    for j in range(b):
        if(binarized[i][j] == 0):
            tmp = (i, j)
            dataset.append(tmp)

#print 'number of data is', len(dataset)

pca = PCA(n_components=1)
newData=pca.fit_transform(dataset)
info_remained =  (pca.explained_variance_ratio_)[0]

min_p = np.min(newData)
max_p = np.max(newData)

cover = len(newData)/float(max_p - min_p)

wealths = []

a, b = newData.shape
for i in range(a):
    wealths.append(newData[i][0]-min_p)

gini =  gini_coef(wealths)

print 'var = ', info_remained
print 'gini = ', gini
print 'cover = ', cover
print 'gini*cover = ', gini*cover













