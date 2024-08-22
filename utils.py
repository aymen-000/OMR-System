import cv2 as cv 
import numpy as np 
# Show a stack of images
def showStackedImages(imgArray, title, scale, labels=None, save=None):
    rows = len(imgArray)
    cols = len(imgArray[0])
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]

    # Resize images according to scale and convert grayscale images to BGR
    for i in range(rows):
        for j in range(cols):
            imgArray[i][j] = cv.resize(imgArray[i][j], (0, 0), None, scale, scale)
            if len(imgArray[i][j].shape) == 2:  # if the image is grayscale
                imgArray[i][j] = cv.cvtColor(imgArray[i][j], cv.COLOR_GRAY2BGR)

    # Stack images horizontally
    hor = [np.hstack(imgArray[i]) for i in range(rows)]
    ver = np.vstack(hor)

    # Add labels if provided
    if labels is not None:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        for i in range(rows):
            for j in range(cols):
                cv.rectangle(ver, (j * eachImgWidth, i * eachImgHeight),
                             (j * eachImgWidth + len(labels[i][j]) * 13 + 27, 30 + i * eachImgHeight),
                             (255, 255, 255), cv.FILLED)
                cv.putText(ver, labels[i][j], (j * eachImgWidth + 10, i * eachImgHeight + 20),
                           cv.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)

    # Save the image if save path is provided
    if save is not None:
        cv.imwrite(save, ver)

    # Show the image
    ver = cv.resize(ver , (1500,700))
    cv.imshow(title, ver)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
def RectangleCou(contours): 
    recangles = []
    for i in contours : 
        area = cv.contourArea(i)
        if area > 50 :
            epsilon = 0.06*cv.arcLength(i , True)
            approx = cv.approxPolyDP(i , epsilon , True)
            if len(approx) == 4 : 
                recangles.append(i)
    
    recCont = sorted(recangles , key=cv.contourArea , reverse=True)
    
    return recCont
            
            
def getConrnerPoint(contour) : 
    apperox = cv.approxPolyDP(contour , 0.01*cv.arcLength(contour , True) , True)
    return apperox


def reorder(myPoint) : 
    myPoint = myPoint.reshape((4,2))
    sum = myPoint.sum(1) # get the sum of each x,y 
    newPoint = np.zeros((4,1,2))
    newPoint[0] = myPoint[np.argmin(sum)]
    newPoint[3] = myPoint[np.argmax(sum)]
    diff = np.diff(myPoint , axis=1)
    newPoint[1] = myPoint[np.argmin(diff)]
    newPoint[2] = myPoint[np.argmax(diff)]
    
    return newPoint
   
def split_img(img , questions , choices) : 
    rows = np.vsplit(img , questions) 
    boxes = []
    for r in rows : 
        cols = np.hsplit(r , choices)
        for c in cols : 
            boxes.append(c) 
    return boxes 

def get_all_zero_count(boxes , choices , questions) : 
    # Initialize the zeroPixels array
    zeroPixels = np.zeros((choices , questions))

    # Initialize row index
    i = 0

    # Loop through the boxes and count non-zero pixels
    for idx, img in enumerate(boxes):
        # Calculate the number of non-zero pixels
        nonZero = cv.countNonZero(img)

        # Calculate the column index based on idx
        j = idx % 5

        # Store the non-zero count in the appropriate location in zeroPixels
        zeroPixels[i][j] = nonZero

        # Increment the row index every 5 images
        if j == 4:
            i += 1
    return zeroPixels

def get_answers(pixels) : 
    ans = [] 
    for row in  pixels : 
        maxIndex = np.argmax(row)
        ans.append(maxIndex)
    return ans

def get_final_score(right , student) : 
    score = 0
    target_score= [] # score for each row
    for item1 , item2 in zip(right , student) : 
        if item1 == item2 :
            score = score +1
            target_score.append(1)
        else : 
            target_score.append(0)
    return (score / len(student) , target_score )

def show_answers(img , questions , choices , Index , grading , ans ) : 
    w , h = int(img.shape[1]/choices) , int(img.shape[0]/questions)
    for x in range(questions) : 
        myAns = Index[x]
        cx =(myAns*w) + w//2
        cy = x*h + h//2
        if grading[x] == 1 : 
            color = (0,255,0)
        else : 
            color = (0,0,255)
            cv.circle(img , ((ans[x]*w) + w//2, cy) , 40 , (0,255,0) , cv.FILLED)
        cv.circle(img , (cx , cy) , 40 , color , cv.FILLED)

    return img