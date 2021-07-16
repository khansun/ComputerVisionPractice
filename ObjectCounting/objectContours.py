import cv2 as cv
import numpy as np


def drawAllContours(img, blurKernel_size=5, morphKernel_size=5, canny_config=(45,180)):
    img = cv.resize(img, (600,600))
    img_out = img
    kernel_morph = np.ones((morphKernel_size, morphKernel_size))
    kernel_blur = (blurKernel_size, blurKernel_size)
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray = cv.GaussianBlur(img_gray, kernel_blur, 1)
    
    img_gray = cv.erode(img_gray, kernel_morph, iterations=2)
    img_gray = cv.dilate(img_gray, kernel_morph, iterations=2)
    
    img_gray = cv.Canny(img_gray, canny_config[0], canny_config[1], 3)
    contours, hierarchy = cv.findContours(img_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img_out, contours, -1, (0,255,0), 1)
    count = len(contours)
    return img_out, count
    

def main(gridPath):
    img = cv.imread(gridPath)
    img_out, count = drawAllContours(img,11)
    labelTextX = "Objects Detected: "
    labelTextY = str(count)
    lengthLabel = cv.putText(img_out, (labelTextX+labelTextY),(1,30), cv.FONT_ITALIC, 1, (255,0,0), 2, cv.LINE_AA,False)
    cv.imshow('Output' , img_out)
    cv.imshow('Input' , img)
    cv.waitKey(0)
    cv.destroyAllWindows() 


if __name__ == "__main__": 
    
    gridPath = "./img_repo/wall.jpg"
    main(gridPath) 

    # Filters are to be tuned according to the image