import cv2 as cv
import numpy as np
import re

def getContourPoints(img, cannyHigh=100, cannyLow=50, minArea=1000, corners=0, draw =False):
    grayImg = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    gaussImg = cv.GaussianBlur(grayImg,(5,5),1)
    cannyImg = cv.Canny(gaussImg,cannyHigh,cannyLow,apertureSize=3)
    mask = np.ones((5,5))
    dilatedImg = cv.dilate(cannyImg,mask,iterations=2)
    closedImg = cv.erode(dilatedImg,mask,iterations=2)
    contours,hierarchy = cv.findContours(closedImg,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    outContours = []
    
    for i in contours:
        area = cv.contourArea(i)
        if area > minArea:
            perimeter = cv.arcLength(i,True)
            approx = cv.approxPolyDP(i,0.02*perimeter,True)
            boundRect = cv.boundingRect(approx)
            if corners > 0:
                if len(approx) == corners:
                    outContours.append([len(approx),area,approx,boundRect,i])
            else:
                outContours.append([len(approx),area,approx,boundRect,i])
    
    outContours = sorted(outContours,key = lambda x:x[1] ,reverse= True)
    if draw:
        for con in outContours:
            cv.drawContours(img,con[4],-1,(255,0,0),5)
    return outContours, hierarchy

def shiftCorners(points):
    newPoints = np.zeros_like(points)
    shapedPoints = points.reshape(4,2)
    sum  = shapedPoints.sum(1)
    newPoints[0] = shapedPoints[np.argmin(sum)]
    newPoints[3] = shapedPoints[np.argmax(sum)]
    sliceAxis = np.diff(shapedPoints,axis=1)
    newPoints[1] = shapedPoints[np.argmin(sliceAxis)]
    newPoints[2] = shapedPoints[np.argmax(sliceAxis)]
    return newPoints

def imgFlex(img, boundPoints, xLen, yLen, refinePeri=20):
    boundPoints = shiftCorners(boundPoints)
    scaleRect = np.float32([[0, 0], [xLen, 0], [0, yLen], [xLen,yLen]])
    fullRect = np.float32(boundPoints)
    matTransform = cv.getPerspectiveTransform(fullRect,scaleRect)
    flexImg = cv.warpPerspective(img, matTransform, (xLen,yLen))
    flexImg = flexImg[refinePeri: flexImg.shape[0]-refinePeri, refinePeri: flexImg.shape[1]-refinePeri]
    return flexImg
    
def euclidDistance(pointA, pointB):
    return np.sqrt((pointA[0]-pointB[0])**2 + (pointA[1]-pointB[1])**2)


def main(path, scaleXlen, scaleYlen):
    scalar = 4
    img = cv.imread(path)
    contourPoints, _ = getContourPoints(img, minArea=1920, corners=4)
    for dots in contourPoints:
        cv.polylines(img, [dots[2]], True, (0,255,0),2)
    

    if(len(contourPoints)>0):
        largestContour = contourPoints[0][2]
        flexImg = imgFlex(img, largestContour,scaleXlen*scalar,scaleYlen*scalar)
        targetContourPoints, _ = getContourPoints(flexImg, minArea=128, corners=4, draw=True)
        #print(targetContourPoints)
        if (len(targetContourPoints)>0):

            for dots in targetContourPoints:
                cv.polylines(flexImg, [dots[2]], True, (0,0,255),1)
                targetCorners = shiftCorners(dots[2])
            
            peakDistanceX = round(euclidDistance(targetCorners[0][0]//scalar, targetCorners[1][0]//scalar),2)
            peakDistanceY = round(euclidDistance(targetCorners[0][0]//scalar, targetCorners[2][0]//scalar),2)
    
            
            labelTextX = "Length: Horizontal = "+str(peakDistanceX)+"mm; "
            labelTextY = "Vertical = "+str(peakDistanceY) +"mm"
            lengthLabel = cv.putText(flexImg, (labelTextX+labelTextY),(10,50), cv.FONT_ITALIC, 1, (0,0,0), 1, cv.LINE_AA,False)

            cv.imshow('Target Area',cv.resize(flexImg, (800,600), interpolation = cv.INTER_AREA))
            

    cv.waitKey(0)
    cv.destroyAllWindows()
    return flexImg  


if __name__ == "__main__": 

    path = "ImageLibrary/proArt.jpg"
    paperXmm = 291
    paperYmm = 229
    outImg = main(path, paperXmm, paperYmm)
    cv.imwrite(re.sub(".jpg", "_area.jpg", path), outImg) 

