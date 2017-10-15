import numpy as np
from nnmat import *
import os

import sys
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import meshandler
 
import random
import cv2

class ImgNN:
    def __init__(self, shape, resultShape = (16, 16), imageSize = (400,400)):
        self.resultShape = resultShape
        self.w = imageSize[0] // shape[0]
        self.h = imageSize[1] // shape[1]
        self.net = NN([shape, (1,shape[0]), (1,1)])
        self.shape = shape
        self.imageSize = imageSize

    def learn(self, srcArr, result, cycles):
        for c in range(cycles):
            for x in range(self.w):
                for y in range(self.h):
                    a = srcArr[x:x+self.shape[0], y:y+self.shape[1]]
                    if a.shape != (self.shape[0], self.shape[1]):
                        print(a.shape)
                        continue
                    self.net.learn(a, result[x,y], 1)

    def calc(self, srcArr):
        resArr = np.zeros(self.resultShape)
        for x in range(self.w):
            for y in range(self.h):
                a = srcArr[x:x+self.shape[0], y:y+self.shape[1]]
                if a.shape != (self.shape[0], self.shape[1]):
                    continue
                if x >= self.resultShape[0] or y >= self.resultShape[1]:
                    continue
                res = self.net.calc(a)
                resArr[x,y] = res[0,0]
        return resArr
        
    def learnFile(self, file, result, cycles):
        return self.learn(readImage(file, self.imageSize), result, cycles)

    def calcFile(self, file):
        return self.calc(readImage(file, self.imageSize))

def readImageCV(file, imageSize):
    img = cv2.imread(file)
    small = cv2.resize(img, imageSize)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    return hsv[:,:,0]/255

def readImageQ(file, imageSize):
    img = QImage(file)
    if img.isNull():
        return 0
    img = img.convertToFormat(QImage.Format_Grayscale8)
    img = img.scaled(imageSize[0],imageSize[1],Qt.IgnoreAspectRatio)
    srcBi = img.bits()
    srcBi.setsize(img.width() * img.height())
    srcBy = bytes(srcBi)

    srcW, srcH = img.width(), img.height()
    srcArr = np.recarray((srcH, srcW), dtype=np.uint8, buf=srcBy).view(dtype=np.uint8,type=np.ndarray)
    return srcArr/255

def readImageCVQ(file, imageSize):
    img = QImage(file)
    if img.isNull():
        return 0
    img = img.convertToFormat(QImage.Format_RGB888)
    img = img.scaled(imageSize[0],imageSize[1],Qt.IgnoreAspectRatio)
    srcBi = img.bits()
    srcBi.setsize(img.byteCount())
    srcBy = bytes(srcBi)

    srcW, srcH = img.width(), img.height()
    bp = img.depth() // 8
    srcArr = np.recarray((srcH, srcW, bp), dtype=np.uint8, buf=srcBy)
    srcArr = srcArr.view(dtype=np.uint8,type=np.ndarray)
##    srcArr = np.uint8(srcArr)   
##    small = srcArr
    hsv = cv2.cvtColor(srcArr, cv2.COLOR_RGB2HSV)
    return hsv[:,:,0]/255


if __name__ == '__main__':

    readImage = readImageCVQ

##    i = readImage('C:\\Users\\ВС\\Pictures\\habr3\\flowers.png',(960,480))
##    cv2.imshow("i",i)
##    print(i)
##else:
    y = np.array([[1,0,1,0]])
    firstShape = (40, 40)
    middleShape = (10, 10)
    imageSize = firstShape[0]*middleShape[0], firstShape[1]*middleShape[1]
    

    StartLearn = False

    if not StartLearn:
        pictDir = 'C:\\Users\\ВС\\Pictures\\Отобранные' # '2014-05'
        nn = ImgNN(firstShape, resultShape=middleShape, imageSize=imageSize)
##        print(nn.net.syns[0].shape,nn.net.syns[1].shape)
        nn.net.syns[0] = np.loadtxt('syns_save0.txt',ndmin=nn.net.syns[0].ndim)
        nn.net.syns[1] = np.loadtxt('syns_save1.txt',ndmin=nn.net.syns[1].ndim)
##        print(nn.net.syns[0].shape,nn.net.syns[1].shape)
        nn2 = NN([middleShape, (y.shape[1], middleShape[0]), y.shape])
##        print(nn2.syns[0].shape,nn2.syns[1].shape)
        nn2.syns[0] = np.loadtxt('syns2_save0.txt',ndmin=nn2.syns[0].ndim)
        nn2.syns[1] = np.loadtxt('syns2_save1.txt',ndmin=nn2.syns[1].ndim)
##        print(nn2.syns[0].shape,nn2.syns[1].shape)
        files = [e.path for e in os.scandir(pictDir)]
        for f in files:
            i = readImage(f, imageSize)
            mid = nn.calc(i)
            res = nn2.calc(mid)
            delta = y-res
            v = round(np.std(delta),3)
            if v <= 0.3:
                print('Flower',f,v)
##            else:
##                print('No flower',f, v)
    else:    
        fl = [e.path for e in os.scandir('flowers')]
        nofl = [e.path for e in os.scandir('noflowers')]
        all = fl+nofl
        yy = np.zeros(middleShape)
        np.fill_diagonal(yy,1)
        minFails = None
        lastSyns = None
        nextSyns = None
        lastSyns2 = None
        lastYY = yy
        nextYY = yy
        minDy = None
        maxDn = None
        for epoch in range(100):
            print('Epoch =', epoch)
            nn = ImgNN(firstShape, resultShape=middleShape, imageSize=imageSize)
            if not (lastSyns is None):
                nextSyns = lastSyns
                for r in range(len(nextSyns)):
                    rand = (np.random.random(nextSyns[r].shape)-0.5)/20
                    nextSyns[r] = nextSyns[r] + rand
                nn.net.syns = nextSyns
            nn2 = NN([middleShape, (y.shape[1], middleShape[0]), y.shape])
            for f in fl:
                i = readImage(f, imageSize)
                nn.learn(i, nextYY, 2)
##                nn.learn(i, yy, 2)
                mid = nn.calc(i)
                nn2.learn(mid, y, 1000)
            nextSyns=None
            fails = 0
            failFiles = []
            dy = 0.0
            dn = 0.0
            for f in all:
                i = readImage(f, imageSize)
                mid = nn.calc(i)
                res = nn2.calc(mid)
                delta = (y-res)
                v = round(np.std(delta),3)
                #v = round(delta.sum(),3)
                print(f, 'res = ', res.round(2),'v =',v)
                if f in fl:
                    dy += v
                if f in nofl:
                    dn += v
                if v > 0.2 and f in fl:
                    fails += 1
                    failFiles.append(f)
                elif v<0.2 and f in nofl:
                    fails +=1
                    failFiles.append(f)
            print('dy =',dy,'dn =',dn)
            if minDy == None or dy < minDy:
                minDy = dy
##                lastYY = newYY
##                lastSyns = nn.net.syns
##                lastSyns2 = nn2.syns
            if maxDn == None or dn > maxDn:
                maxDn = dn
            if minFails == None or fails < minFails:
                minFails = fails
                lastSyns = nn.net.syns
                lastSyns2 = nn2.syns
                lastYY = nextYY
            else:
                nextYY = lastYY +(np.random.random(yy.shape)-0.5)/20
            print('fails =',fails, failFiles)
            print('min =',minFails)
            if minFails <= 1:
                print('found!')
                break
        for i in range(len(lastSyns)):
            np.savetxt('syns_save%s.txt'%i, lastSyns[i])
        for i in range(len(lastSyns2)):
            np.savetxt('syns2_save%s.txt'%i, lastSyns2[i])
