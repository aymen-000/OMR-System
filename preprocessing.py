import cv2 as cv 
import numpy as np 
import utils
def preprocessing(img_path , width , height) : 
    img = cv.imread(img_path)
    img = cv.resize(img , (width  , height))

    # get gray scale
    imgGray = cv.cvtColor(img ,cv.COLOR_BGR2GRAY)
    # blur image 
    imgBlur = cv.blur(imgGray , (3,3) , 1)
    # image canny 
    imgCanny = cv.Canny(imgBlur , 10 , 50)
    # get contours 
    contours , hirarchy = cv.findContours(imgCanny , cv.RETR_EXTERNAL  , cv.CHAIN_APPROX_NONE)
    
    return img , imgGray , imgBlur , imgCanny , contours
    

def find_recatngles(contours , img , height , width) : 
    recCont = utils.RectangleCou(contours=contours)
    AnswerPoints = utils.getConrnerPoint(recCont[0])
    GradePoint = utils.getConrnerPoint(recCont[1])
    if AnswerPoints.size != 0 and GradePoint.size != 0 :
        BiggestContour , SecondBiggestContour = np.float32(utils.reorder(AnswerPoints)) , np.float32(utils.reorder(GradePoint))
        pt2 = np.float32([[0,0], [width , 0] , [0 , height] , [width , height]])
        matrix = cv.getPerspectiveTransform(BiggestContour , pt2)
        
        # get just the answer part (box)
        imgWarpped = cv.warpPerspective(img , matrix ,(width , height))
        
        # for the grading box 
        pt3 = np.float32([[0,0], [width, 0] , [0 , height] , [width , height]])
        matrix2 = cv.getPerspectiveTransform(SecondBiggestContour , pt3)
        gradeWarpped = cv.warpPerspective(img , matrix2 , (width , height))
        invMatrix = cv.getPerspectiveTransform(pt2,BiggestContour)
        invGradePerspectiveMertix = cv.getPerspectiveTransform(pt3, SecondBiggestContour) 
        return imgWarpped , gradeWarpped , invMatrix , invGradePerspectiveMertix

def get_final_score(imgWrapped , questions , choices , right_answers) : 
    imgWarppedGray = cv.cvtColor(imgWrapped , cv.COLOR_BGR2GRAY)
    imgThres = cv.threshold(imgWarppedGray  , 150 , 255 , cv.THRESH_BINARY_INV)[1]
    
    # split the answers to boxes 
    boxes = utils.split_img(imgThres , questions=questions , choices=choices)
    zeroPixels = utils.get_all_zero_count(boxes=boxes ,choices  =choices , questions = questions)
    # find answers
    
    ans = utils.get_answers(zeroPixels)
    # get final score 
    score , score_for_each = utils.get_final_score(right_answers , ans)
    score = score * 100 
    
    return score , score_for_each , ans
         