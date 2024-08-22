import cv2 as cv 
import numpy as np 
import utils

def get_final_results(img , imgWarpped , score , right_answers , score_for_each , ans , file_to_save  , questions , choices , width , height , invMatrix , invGradePerspectiveMertix , gradeWarpped) :
    imgWarppedCopy = imgWarpped.copy()
    new_img_show = utils.show_answers(imgWarppedCopy , questions , choices , ans , score_for_each , right_answers)
    # get black img 
    imgDraw = np.zeros_like(imgWarpped)
    imgDrawBoxes = utils.show_answers(imgDraw , 5 , 5 , ans , score_for_each , right_answers)
    # apply inverse wrapped 
    invWarppedImg = cv.warpPerspective(imgDrawBoxes , invMatrix , (width , height))
    # cshow the score 
    imgFinal = img.copy()
    if score < 50.0 : 
        color = (45,34,255) 
    else : 
        color = (0,255,255)
    gradeLike = np.zeros_like(gradeWarpped)
    gradeLike = cv.putText(gradeLike , f"{str(score)}%" , (30,150) , 3 , cv.FONT_HERSHEY_COMPLEX , color  , 20)
    invGradeImg = cv.warpPerspective(gradeLike , invGradePerspectiveMertix , (width ,height))
    imgFinal = cv.addWeighted(imgFinal , 1 , invWarppedImg , 1 ,0)
    
    imgFinal = cv.addWeighted(imgFinal , 1 , invGradeImg , 3,0)
    
    cv.imwrite(file_to_save , imgFinal)
    return imgFinal
    
    