import cv2 as cv
import numpy as np

def main(gridPath, tilePath):
    img = cv.imread(gridPath)
    img_rgb = img
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_tile = cv.imread(tilePath, 0)
    img = cv.fastNlMeansDenoising(img)
    img = cv.GaussianBlur(img,(5,5),1)
    img_tile = cv.fastNlMeansDenoising(img_tile)
    img_tile = cv.GaussianBlur(img_tile,(5,5),1)
    w, h = img_tile.shape[::-1]
    matchConv = cv.matchTemplate(img, img_tile, cv.TM_CCOEFF_NORMED)
    threshold = .9
    loc = np.where(matchConv >= threshold)
    count = 0
    
    for pt in zip (*loc[::-1]):
        cv.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,0), 1)
        count = count + 1
        
    labelTextX = "Rectangles Drawn: "
    labelTextY = str(count)
    lengthLabel = cv.putText(img_rgb, (labelTextX+labelTextY),(50,50), cv.FONT_ITALIC, 2, (0,0,255), 2, cv.LINE_AA,False)
    cv.imshow('Total Tiles' , cv.resize(img_rgb, (int(img_rgb.shape[1]*.8),int(img_rgb.shape[0]*.8)))) 
    #cv.imshow("Output", img)
    #cv.imshow("Template", img_tile)
    cv.waitKey(0)
    cv.destroyAllWindows() 

if __name__ == "__main__": 

    gridPath = "./img_repo/wall.jpg"
    tilePath = "./img_repo/wallT.jpg"
    main(gridPath, tilePath) 