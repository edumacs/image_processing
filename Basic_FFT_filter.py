import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
import numpy as np
import numpy.ma as ma
from PIL import Image

# --- IMPORT DATA From CSV File

imgFile = "C:/Users/M S I/downloads/eco wares.jpg"
imgPil = Image.open(imgFile).convert('LA')
imgNp = np.array(imgPil.convert('L'))
imgNp = imgNp / np.max(imgNp)
ySize, xSize = imgNp.shape
hafY, hafX = int(ySize / 2), int(xSize / 2)

winXSize = 12
winYSize = 6
winAspect = winXSize / winYSize
plt.close('all')

fig = plt.figure(figsize=(winXSize, winYSize))
fig.canvas.set_window_title('Fourier Interactive')

axOrig = fig.add_axes([.05, .2, .8 / winAspect, .7])
axOrig.axes.set_xticks([])
axOrig.axes.set_yticks([])
axOrig.set_title('Magnitude')
plt.imshow(imgPil, cmap='gray')

# Fourier Transform
fourImg = np.fft.fft2(imgNp)
fourShift = np.fft.fftshift(fourImg)

# Fourier Filtering ####
yy, xx = np.mgrid[-hafY:hafY, -hafX:hafX]
distImg = np.sqrt(xx ** 2 + yy ** 2)

angleImg = np.arctan2(yy, xx)
angleImgFlip = np.fliplr(np.flipud(angleImg))

rad1 = .0
rad2 = hafX/2
angle = .0
angleThresh = -1.
maskR1 = (distImg > rad1)
maskR2 = (distImg < rad2)
maskRadial = np.logical_and(maskR1, maskR2)
maskAngle = (np.sin(angleImg * 2. + angle) >= angleThresh)
maskImg = np.logical_and(maskAngle, maskRadial)
maskImg[hafY, hafX] = True
xmask = ma.make_mask(maskImg)
filtImg = fourShift * xmask

# Inverse Fourier Transform
fourIshft = np.fft.ifftshift(filtImg)
fourInv = np.fft.ifft2(fourIshft)
fourAbs = np.abs(fourInv)

# Plot the inverse FT
axFour = fig.add_axes([.5, .2, .8 / winAspect, .7])
axFour.axes.set_xticks([])
axFour.axes.set_yticks([])
axFour.set_title('Magnitude')
plt.imshow(fourAbs, cmap='gray')

# Show image
plt.show()
